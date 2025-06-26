# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llmcord is a Discord bot that transforms Discord into a collaborative LLM frontend. The entire bot is intentionally implemented as a single Python file (~300 lines) following a minimalist design philosophy.

## Core Architecture

**Single-File Design**: The entire bot logic is contained in `llmcord.py`. This design choice prioritizes simplicity and maintainability over traditional modularity.

**Key Components**:
- **Message Processing**: Reply-based conversation system that builds message chains by following Discord reply references
- **Model Integration**: OpenAI-compatible API support with dynamic model switching via `/model` slash command
- **MCP Tool Calling**: Model Context Protocol integration for web search and scraping capabilities
- **Streaming Responses**: Real-time message updates with Discord embed system
- **Permission System**: Granular user/role/channel-based access control

**Message Chain Architecture**: The bot builds conversation context by traversing Discord reply chains backward, creating a linked list of messages. Each message is cached in `msg_nodes` dict with async locks for thread safety.

**Two-Phase Tool Execution**:
1. **Phase 1**: Stream response to detect tool calls, show progress indicators
2. **Phase 2**: Execute tools with fresh MCP connections, then get final LLM response and edit the progress message

**MCP Integration**: Uses fresh connections per tool call (not cached) due to asyncio context manager requirements. Tools are discovered at startup and mapped to their respective servers in `tool_to_server` dict.

## Development Commands

**Run with Docker** (recommended):
```bash
sudo docker-compose -p llmcord-test up --build
```

**View logs**:
```bash
sudo docker-compose -p llmcord-test logs --tail 50 -f
```

**Restart after code changes** (no rebuild needed):
```bash
sudo docker-compose -p llmcord-test restart
```

**Note**: The container auto-reloads Python code changes, so `restart` is sufficient for most development iterations without needing `--build`.

**Run without Docker**:
```bash
python -m pip install -U -r requirements.txt
python llmcord.py
```

## Configuration

**Required Setup**:
1. Copy `config-example.yaml` to `config.yaml`
2. Configure Discord bot token and client ID from Discord Developer Portal
3. Add LLM provider API keys and model configurations
4. Optional: Configure MCP servers for tool calling (requires Node.js in container)

**MCP Servers**: Configured in `mcp_servers` section of config.yaml. Each server runs as an external process via `npx` and communicates over stdio. Examples included for Brave Search and Firecrawl.

## Code Style and Patterns

**Async/Await**: Fully async codebase using discord.py and asyncio
**Error Handling**: Comprehensive logging with emoji-based log levels for easy scanning
**Memory Management**: Fixed-size `msg_nodes` cache with automatic cleanup
**Global State**: Minimal global variables for configuration, tools, and message cache

**Key Functions**:
- `on_message()`: Main message handler (~200 lines, handles entire conversation flow)
- `execute_mcp_tool()`: Tool execution with proper MCP connection lifecycle
- `model_command()`: Slash command for model switching with autocomplete

## Testing and Debugging

The bot includes extensive logging for debugging tool execution issues. Look for emoji-prefixed log messages to trace execution flow:
- 🔧 Tool execution steps
- 📡 Server routing and connections  
- 🚀 API calls and timing
- ⚠️ Errors and warnings
- ✅ Successful operations

Common issues involve MCP server connectivity and async context manager lifecycle - always use fresh connections per tool call.