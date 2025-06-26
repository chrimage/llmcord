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
import spacy
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


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


async def execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Execute MCP tool by connecting to the appropriate server"""
    # Find which server has this tool
    server_name = tool_to_server.get(tool_name)
    if not server_name:
        return f"Unknown tool: {tool_name}"
    
    mcp_config = config.get("mcp_servers", {})
    server_config = mcp_config.get(server_name)
    if not server_config:
        return f"Server configuration not found: {server_name}"
    
    try:
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"], 
            env=server_config.get("env")
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                
                if result.content and len(result.content) > 0 and hasattr(result.content[0], 'text'):
                    return result.content[0].text
                    
    except Exception:
        logging.exception(f"Failed to execute tool {tool_name} on server {server_name}")
        
    return f"Tool execution failed: {tool_name} on server {server_name}"


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
mcp_tools = []
tool_to_server = {}  # Map tool names to their server names

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("Spacy model 'en_core_web_sm' not found. Please download it by running: python -m spacy download en_core_web_sm")
    # Potentially exit or disable functionality that depends on SpaCy
    nlp = None


def split_text_into_chunks(text: str, max_length: int) -> list[str]:
    """Splits text into chunks by sentences, respecting max_length."""
    if not text:
        return []
    if nlp is None: # Fallback if SpaCy model failed to load
        logging.warning("SpaCy model not loaded. Falling back to basic splitting.")
        # Basic fallback: split by paragraphs, then by length if necessary.
        # This won't be as good as sentence splitting but prevents errors.
        chunks = []
        current_chunk = ""
        for paragraph in text.split('\n\n'):
            if len(current_chunk) + len(paragraph) + 2 > max_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk += (paragraph + "\n\n")

            # If a single paragraph is too long, split it harshly.
            while len(current_chunk) > max_length:
                split_point = current_chunk.rfind(' ', 0, max_length)
                if split_point == -1: # No space found, hard break
                    split_point = max_length
                chunks.append(current_chunk[:split_point])
                current_chunk = current_chunk[split_point:].lstrip()

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        if len(current_chunk) + len(sentence_text) + 1 > max_length: # +1 for potential space
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If a single sentence is longer than max_length, it becomes its own chunk (and might be truncated by Discord later)
            if len(sentence_text) > max_length:
                # This single sentence is too long. We'll add it as is.
                # Discord will handle the hard truncation.
                # Or, we could try to split it further, but that risks breaking mid-word without more complex logic.
                chunks.append(sentence_text)
                current_chunk = ""
            else:
                current_chunk = sentence_text
        else:
            if current_chunk:
                current_chunk += " " + sentence_text
            else:
                current_chunk = sentence_text

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


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
    mcp_config = config.get("mcp_servers", {})
    for server_name, server_config in mcp_config.items():
        try:
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env")
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    
                    # Convert MCP tools to OpenAI function calling format
                    for tool in tools_response.tools:
                        tool_to_server[tool.name] = server_name  # Track which server has which tool
                        mcp_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema
                            }
                        })
                    
                    logging.info(f"Loaded {len(tools_response.tools)} tools from MCP server '{server_name}'")
                    
        except Exception:
            logging.exception(f"Failed to initialize MCP server '{server_name}'")


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
                        current_tool_call = {
                            "id": tool_call_delta.id,
                            "name": tool_call_delta.function.name if tool_call_delta.function else None
                        }
                        tool_arguments_str = ""
                        
                        # Show progress
                        if current_tool_call["name"]:
                            embed.description = f"🔍 Using {current_tool_call['name']}..."
                            embed.color = EMBED_COLOR_INCOMPLETE
                            
                            reply_to_msg = new_msg
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)
                            
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                    
                    # Accumulate function name
                    if tool_call_delta.function and tool_call_delta.function.name:
                        current_tool_call["name"] = tool_call_delta.function.name
                    
                    # Accumulate arguments
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        tool_arguments_str += tool_call_delta.function.arguments
                
                # Tool call complete
                if choice.finish_reason and current_tool_call:
                    try:
                        tool_arguments = json.loads(tool_arguments_str) if tool_arguments_str else {}
                        logging.info(f"Executing tool: {current_tool_call['name']} with args: {tool_arguments}")
                        tool_result = await execute_mcp_tool(current_tool_call["name"], tool_arguments)
                        logging.info(f"Tool result length: {len(tool_result)} chars - Content: {tool_result[:200]}")
                        
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
                        
                    except Exception:
                        logging.exception("Tool execution failed")
                        embed.description = "⚠️ Tool execution failed, continuing..."
                        embed.color = EMBED_COLOR_INCOMPLETE
                        await response_msgs[-1].edit(embed=embed)
                    
                    break
            
            # PHASE 2: Get actual response (either direct response or after tool execution)
            if tool_calls_made:
                # Update params with tool results and make new request
                openai_params["messages"] = messages[::-1]
                openai_params.pop("tools", None)  # Remove tools for second call
                openai_params.pop("tool_choice", None)
                
                # Get the full response (non-streaming) and directly edit the tool progress message
                openai_params["stream"] = False
                response = await openai_client.chat.completions.create(**openai_params)
                final_content = response.choices[0].message.content or "" # Ensure it's a string
                
                # Split final_content into chunks using the new function
                # max_message_length for embeds is 4096, but description has a limit of 4096.
                # We'll use a slightly smaller value for safety and to account for STREAMING_INDICATOR if it were used (though not here).
                # For tool responses, we send full messages, so STREAMING_INDICATOR isn't relevant for the split length.
                # Discord embed description limit is 4096.
                message_chunks = split_text_into_chunks(final_content, 4000) # Use 4000 as a safe limit for embed descriptions

                if not message_chunks: # Handle case where final_content is empty or only whitespace
                    message_chunks = [""]

                # Edit the first message (tool progress message) with the first chunk
                embed.description = message_chunks[0]
                embed.color = EMBED_COLOR_COMPLETE
                await response_msgs[-1].edit(embed=embed)
                
                # Send subsequent chunks as new messages
                for i in range(1, len(message_chunks)):
                    new_embed = discord.Embed(description=message_chunks[i], color=EMBED_COLOR_COMPLETE)
                    reply_to_msg = response_msgs[-1] # Reply to the previous bot message
                    new_response_msg = await reply_to_msg.reply(embed=new_embed, silent=True)
                    response_msgs.append(new_response_msg)
                    msg_nodes[new_response_msg.id] = MsgNode(parent_msg=new_msg) # Parent is still original user message
                    # Lock is not strictly needed here as it's not being streamed/edited further by this part of the code

                # Update response contents for msg_nodes (store the full, unsplit content)
                response_contents = [final_content] # Storing the original full content for the node
            else:
                # Stream the actual response (only for non-tool responses)
                finish_reason = None
                stream_buffer = "" # Accumulates deltas for current processing pass
                accumulated_stream_content = "" # Stores the entire response from the stream

                async for curr_chunk in await openai_client.chat.completions.create(**openai_params):
                    if choice := curr_chunk.choices[0] if curr_chunk.choices else None:
                        delta_content = choice.delta.content or ""
                        stream_buffer += delta_content
                        accumulated_stream_content += delta_content
                        new_finish_reason = choice.finish_reason

                        if not use_plain_responses:
                            # Determine if it's time to process the buffer for sending chunks
                            # Process if: stream is ending, or buffer is getting large enough to potentially form a sentence.
                            # The max_message_length here is the one for embeds (4096 - indicator len)
                            process_now = new_finish_reason or (len(stream_buffer) >= max_message_length / 2 and '\n' in stream_buffer) or len(stream_buffer) >= max_message_length

                            if process_now:
                                # Pass the current buffer to the splitter.
                                # The splitter returns complete sentences or oversized sentences as chunks.
                                chunks_from_buffer = split_text_into_chunks(stream_buffer, max_message_length - len(STREAMING_INDICATOR))

                                # The new stream_buffer will be what's left after the text that formed chunks_from_buffer.
                                # If the stream is not finished, the last chunk returned by splitter might be partial if it had to force split a long sentence,
                                # or if the original buffer ended mid-sentence.
                                # A robust way to find the remaining buffer:
                                consumed_text_len = 0
                                if chunks_from_buffer:
                                    # This assumes split_text_into_chunks doesn't drastically reformat (like adding many newlines)
                                    # It's an approximation of how much of the start of stream_buffer was used.
                                    # A truly robust way is for split_text_into_chunks to also return the length of consumed input.
                                    # For now, let's assume chunks_from_buffer[0]...chunks_from_buffer[N] are contiguous from stream_buffer start.
                                    # This is a simplification:
                                    temp_doc = nlp(stream_buffer)
                                    current_pos = 0
                                    num_sents_in_chunks = 0

                                    # Try to align sentences from nlp doc with chunks_from_buffer
                                    # This is heuristic and might not be perfect.
                                    temp_chunks_to_send = []
                                    temp_stream_buffer_after_processing = stream_buffer

                                    processed_chunk_text_for_this_iteration = []

                                    if new_finish_reason is None and chunks_from_buffer:
                                        # If stream not finished, hold back the content of the last chunk from split_text_into_chunks.
                                        # That content becomes the new stream_buffer.
                                        # All other chunks are sent.
                                        temp_stream_buffer_after_processing = chunks_from_buffer.pop() # Last part is new buffer
                                        processed_chunk_text_for_this_iteration.extend(chunks_from_buffer)
                                    elif chunks_from_buffer: # Stream is finished or no chunks to hold back
                                        processed_chunk_text_for_this_iteration.extend(chunks_from_buffer)
                                        temp_stream_buffer_after_processing = "" # All processed

                                    stream_buffer = temp_stream_buffer_after_processing


                                for i, text_to_embed in enumerate(processed_chunk_text_for_this_iteration):
                                    is_last_of_this_batch = (i == len(processed_chunk_text_for_this_iteration) - 1)
                                    # Is this the absolute final piece of the entire stream response?
                                    is_absolute_final = new_finish_reason and is_last_of_this_batch and not stream_buffer

                                    embed_desc_content = text_to_embed
                                    if not is_absolute_final:
                                        embed_desc_content += STREAMING_INDICATOR

                                    embed.description = embed_desc_content
                                    embed.color = EMBED_COLOR_COMPLETE if is_absolute_final else EMBED_COLOR_INCOMPLETE

                                    # Send new message or edit last one
                                    # A new message is needed if no messages yet, or if this is a new chunk (not just an update to current one)
                                    # This simplified logic sends new message for each processed chunk from the list.
                                    if not response_msgs or (response_msgs[-1].embeds and response_msgs[-1].embeds[0].description.removesuffix(STREAMING_INDICATOR) != text_to_embed):
                                        if edit_task: await edit_task
                                        reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                        new_resp_msg = await reply_to_msg.reply(embed=embed, silent=True)
                                        response_msgs.append(new_resp_msg)
                                        msg_nodes[new_resp_msg.id] = MsgNode(parent_msg=new_msg)
                                        await msg_nodes[new_resp_msg.id].lock.acquire()
                                        last_task_time = datetime.now().timestamp()
                                    else: # Edit current message
                                        can_edit_now = (edit_task is None or edit_task.done()) and \
                                                       (datetime.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS)
                                        if can_edit_now or is_absolute_final:
                                            if edit_task: await edit_task
                                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                            last_task_time = datetime.now().timestamp()
                        
                        finish_reason = new_finish_reason # Update overall finish_reason
                        if finish_reason:
                            break # Exit async for loop if stream is done

                # After the streaming loop, if there's any content left in stream_buffer (e.g. stream ended, last part wasn't processed)
                # and we are not using plain responses, send this remainder.
                if not use_plain_responses and stream_buffer:
                    embed.description = stream_buffer # Final part, no streaming indicator
                    embed.color = EMBED_COLOR_COMPLETE
                    if not response_msgs: # If somehow no messages were ever sent
                        reply_to_msg = new_msg
                        new_final_msg = await reply_to_msg.reply(embed=embed, silent=True)
                        response_msgs.append(new_final_msg)
                        msg_nodes[new_final_msg.id] = MsgNode(parent_msg=new_msg)
                        await msg_nodes[new_final_msg.id].lock.acquire()
                    else: # Edit the last message with the final final content
                        if edit_task: await edit_task
                        await response_msgs[-1].edit(embed=embed)

                # Update response_contents to hold the single, complete accumulated string
                # This is used by plain responses and for msg_node text.
                response_contents = [accumulated_stream_content]

            # After streaming or tool call, response_contents should have the full response.
            # If it was a tool call, response_contents was set in that block.
            # If it was streaming, it's set to [accumulated_stream_content] above.
            # Ensure it's a list with at least one string for the join/processing below.
            if not response_contents:
                response_contents = [""]


            if use_plain_responses:
                # If plain responses, split the full text and send each chunk.
                # max_message_length is 2000 for plain text.
                message_chunks = split_text_into_chunks(full_response_text, max_message_length)

                if not message_chunks and not response_msgs: # Ensure at least one empty message if response was empty
                    message_chunks = [""]
                elif not message_chunks and response_msgs: # If there were prior (e.g. embed) messages, don't send empty plain
                    pass


                # Clear existing response_msgs if they were from a different format (e.g. embed stream attempt)
                # This part is tricky because response_msgs might already contain an embed if streaming started that way.
                # For now, we assume if use_plain_responses is true, the streaming logic above for embeds
                # might have created an initial message. We should ensure these are plain.
                # This might need a more robust check or state management if mixing modes.
                # For simplicity, if use_plain_responses is true, we assume all final messages should be plain.

                # Delete any preliminary embed messages if we are now sending plain text
                for msg_to_delete in response_msgs:
                    try:
                        await msg_to_delete.delete()
                    except discord.HTTPException:
                        logging.warning(f"Could not delete preliminary message {msg_to_delete.id} before sending plain text.")
                response_msgs = [] # Reset response_msgs for plain text sending

                for chunk_content in message_chunks:
                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=chunk_content or "\u200b", suppress_embeds=True) # Send ZWS if empty
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    # Lock acquire might not be needed if these messages aren't further manipulated,
                    # but let's keep it for consistency with the original structure for msg_nodes.
                    await msg_nodes[response_msg.id].lock.acquire()

            # If not use_plain_responses and streaming happened, the embed messages are already sent/edited.
            # The full_response_text is stored in msg_nodes below.

    except Exception:
        logging.exception("Error while generating response")

    # Store the full, unsplit text in msg_nodes for all bot responses.
    # The individual chunks are what's sent, but the node should represent the complete logical message.
    full_final_text = "".join(response_contents)
    for response_msg in response_msgs:
        # If this message was part of a chunked response, its .text should ideally be its own chunk.
        # However, the original logic sets msg_nodes[response_msg.id].text to the *entire* response for all messages.
        # We will keep this behavior for now, as changing it might affect conversation history logic.
        msg_nodes[response_msg.id].text = full_final_text
        if msg_nodes[response_msg.id].lock.locked(): # Release lock if acquired
            msg_nodes[response_msg.id].lock.release()

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
