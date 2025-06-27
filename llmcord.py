import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude", "gemini", "gemma", "llama", "pixtral", "mistral", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500
MAX_EMBED_LENGTH = 4000
MCP_CONNECTION_TIMEOUT = 30
MCP_TOOL_TIMEOUT = 60


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


async def execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    server_name = tool_to_server.get(tool_name)
    if not server_name:
        return f"❌ Unknown tool: {tool_name}"
    
    server_config = config.get("mcp_servers", {}).get(server_name)
    if not server_config:
        return f"❌ Server not found: {server_name}"
    
    try:
        server_params = StdioServerParameters(command=server_config["command"], args=server_config["args"], env=server_config.get("env"))
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=MCP_CONNECTION_TIMEOUT)
                result = await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=MCP_TOOL_TIMEOUT)
                return result.content[0].text if result.content and hasattr(result.content[0], 'text') else f"⚠️ {tool_name} returned no content"
    except asyncio.TimeoutError:
        return f"⏱️ {tool_name} timed out"
    except Exception as e:
        return f"⚠️ {tool_name} failed: {str(e)[:50]}"


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
mcp_tools = []
tool_to_server = {}  # Map tool names to their server names
# Removed connection caching to fix async context manager issues

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices


@discord_bot.event
async def on_ready() -> None:
    global mcp_tools
    
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

    await discord_bot.tree.sync()
    
    # Initialize MCP servers and load available tools
    for server_name, server_config in config.get("mcp_servers", {}).items():
        try:
            server_params = StdioServerParameters(command=server_config["command"], args=server_config["args"], env=server_config.get("env"))
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        tool_to_server[tool.name] = server_name
                        mcp_tools.append({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema}})
                    logging.info(f"🔧 Loaded {len(tools_response.tools)} tools from MCP server '{server_name}'")
        except Exception as e:
            logging.error(f"💥 Failed to initialize MCP server '{server_name}': {str(e)[:100]}")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or config["allow_dms"] if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.split("/", 1)
    model_parameters = config["models"].get(provider_slash_model, None)

    base_url = config["providers"][provider]["base_url"]
    api_key = config["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config["max_text"]
    max_images = config["max_images"] if accept_images else 0
    max_messages = config["max_messages"]

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config["system_prompt"]:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []
    
    # Tool call accumulation for streaming tool data
    current_tool_call = None
    tool_arguments_str = ""

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    use_plain_responses = config["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    try:
        async with new_msg.channel.typing():
            # Prepare OpenAI parameters
            openai_params = {
                "model": model, 
                "messages": messages[::-1], 
                "stream": True, 
                "extra_body": model_parameters
            }
            
            # Add tools if available
            if mcp_tools:
                openai_params["tools"] = mcp_tools
                openai_params["tool_choice"] = "auto"
            
            # PHASE 1: Check for tool calls (OpenAI may choose to call tools instead of responding directly)
            tool_calls_made = False
            async for curr_chunk in await openai_client.chat.completions.create(**openai_params):
                if not (choice := curr_chunk.choices[0] if curr_chunk.choices else None):
                    continue

                # If we get content instead of tool calls, break and handle normally
                if choice.delta.content:
                    curr_content = choice.delta.content
                    finish_reason = choice.finish_reason
                    break
                
                # Handle tool calls
                if choice.delta.tool_calls:
                    tool_calls_made = True
                    tool_call_delta = choice.delta.tool_calls[0]
                    
                    # Start new tool call
                    if tool_call_delta.id:
                        current_tool_call = {"id": tool_call_delta.id, "name": tool_call_delta.function.name if tool_call_delta.function else None}
                        tool_arguments_str = ""
                        
                        # Show progress
                        if current_tool_call["name"]:
                            embed.description = f"🔍 Using {current_tool_call['name']}..."
                            embed.color = EMBED_COLOR_INCOMPLETE
                            response_msg = await new_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                    
                    # Accumulate function name and arguments
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            current_tool_call["name"] = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_arguments_str += tool_call_delta.function.arguments
                
                # Tool call complete
                if choice.finish_reason and current_tool_call:
                    try:
                        tool_arguments = json.loads(tool_arguments_str) if tool_arguments_str else {}
                        tool_result = await execute_mcp_tool(current_tool_call["name"], tool_arguments)
                        
                        # Clear progress message - actual response will replace it
                        # Don't update the embed here, let the streaming response take over
                        
                        # Add assistant message with tool call
                        messages.insert(0, {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": current_tool_call["id"],
                                "type": "function", 
                                "function": {
                                    "name": current_tool_call["name"],
                                    "arguments": tool_arguments_str
                                }
                            }]
                        })
                        
                        # Add tool result
                        messages.insert(0, {
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": current_tool_call["id"]
                        })
                        
                    except Exception as e:
                        embed.description = f"⚠️ Tool failed: {str(e)[:50]}..."
                        embed.color = EMBED_COLOR_INCOMPLETE
                        try:
                            await response_msgs[-1].edit(embed=embed)
                        except Exception:
                            pass
                        break
                    
                    break
            
            # PHASE 2: Get actual response (either direct response or after tool execution)
            if tool_calls_made:
                # Update params with tool results and make new request
                openai_params["messages"] = messages[::-1]
                openai_params.pop("tools", None)  # Remove tools for second call
                openai_params.pop("tool_choice", None)
                openai_params["stream"] = False
                
                api_start_time = datetime.now()
                response = await openai_client.chat.completions.create(**openai_params)
                api_time = (datetime.now() - api_start_time).total_seconds()
                
                final_content = response.choices[0].message.content
                
                # Truncate if too long for Discord embed
                if len(final_content) > 4096:
                    final_content = final_content[:4096] + "...\n\n*[Response truncated due to length]*"
                
                embed.description = final_content
                embed.color = EMBED_COLOR_COMPLETE
                await response_msgs[-1].edit(embed=embed)
                
                # Update response contents for msg_nodes
                response_contents = [final_content]
            else:
                # Stream the actual response (only for non-tool responses)
                # Reset curr_content and finish_reason if we fell through from Phase 1 without making a tool call
                curr_content = None
                finish_reason = None
                # Accumulate full response before sending for non-tool call responses
                full_response_content = ""
                async for curr_chunk in await openai_client.chat.completions.create(**openai_params):
                    if not (choice := curr_chunk.choices[0] if curr_chunk.choices else None):
                        continue

                    if choice.delta.content:
                        full_response_content += choice.delta.content

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        break # Ensure we exit once the stream is finished

                if full_response_content:
                    # Split content if it exceeds max_message_length
                    # This logic is simplified; real splitting would need to be more careful about cutting words/markdown
                    temp_content_storage = []
                    while len(full_response_content) > max_message_length:
                        # Find a good split point (e.g., newline or space)
                        split_at = full_response_content.rfind('\n', 0, max_message_length)
                        if split_at == -1: # If no newline, try space
                            split_at = full_response_content.rfind(' ', 0, max_message_length)
                        if split_at == -1 or split_at == 0: # If no good split, force split
                            split_at = max_message_length
                        
                        temp_content_storage.append(full_response_content[:split_at])
                        full_response_content = full_response_content[split_at:].lstrip()
                    temp_content_storage.append(full_response_content) # Add the remainder

                    for content_part in temp_content_storage:
                        if not content_part.strip(): # Avoid sending empty messages
                            continue
                        response_contents.append(content_part) # Store for msg_node later

                        if not use_plain_responses:
                            embed.description = content_part
                            embed.color = EMBED_COLOR_COMPLETE # Assuming completion as it's sent at once

                            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)

                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            # Lock is acquired before adding to msg_nodes in the original, maintaining pattern
                            # but might need release if not handled by the end loop
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            # Handling for use_plain_responses remains, sending content directly
                            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(content=content_part, suppress_embeds=True)
                            response_msgs.append(response_msg)
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()


            # This section handles sending plain responses if use_plain_responses is true
            # AND if the response was NOT streamed via the new logic above (e.g. tool call final response)
            # The new streaming logic for regular messages already handles sending.
            # We only need to ensure that if use_plain_responses is true, and we have response_contents
            # that haven't been sent (e.g. from a non-streamed tool response), they get sent.
            if use_plain_responses and response_contents and not response_msgs: # If content exists but no messages sent yet
                for content_part in response_contents: # response_contents would have been populated by tool response logic
                    if not content_part.strip():
                        continue
                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content_part, suppress_embeds=True)
                    response_msgs.append(response_msg)
                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    # Update msg_nodes with the text and release locks
    # Ensure that text for msg_nodes is correctly assigned from potentially multiple parts
    full_final_text = "".join(response_contents)
    for response_msg in response_msgs:
        # If response_msgs were created progressively for very long messages,
        # this simplistic assignment might not be ideal.
        # However, with the new logic, each response_msg should correspond to a part of the full_final_text.
        # For simplicity, we'll assign the full text to all, or adjust if parts are stored differently.
        # The current response_contents should hold the parts that were actually sent.
        # Let's refine this: each msg_node should get its specific content part.
        # This requires response_contents to accurately reflect what was sent in each message.
        # The modified logic appends to response_contents for each part sent.
        pass # The text is assigned when the message is created/sent in the loop above.
             # We just need to ensure locks are released.

    for response_msg_id in [msg.id for msg in response_msgs]: # Iterate by ID in case response_msgs list is modified
        if response_msg_id in msg_nodes and msg_nodes[response_msg_id].lock.locked():
            # Assign the text to the msg_node. If response_contents has multiple parts,
            # and response_msgs also has multiple, we need to map them.
            # For now, let's assume response_contents holds the full concatenated text if not split,
            # or individual parts if split. The current logic populates response_contents with parts.
            # The text field of MsgNode is used for conversation history.
            # So, the *entire* response should be available.
            msg_nodes[response_msg_id].text = full_final_text # Assign full text for history
            msg_nodes[response_msg_id].lock.release()


    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
