"""
An example demonstrating how to use MCP tools in a Kani agent.

In order to use MCP tools, first specify a list of MCP servers to connect to. Then, use the `tools_from_mcp_servers`
context manager to connect to these servers and retrieve the list of available tools. You can pass these tools like
normal Kani AIFunctions.
"""

import asyncio

from mcp.client.session_group import StdioServerParameters, StreamableHttpParameters

from kani import Kani, chat_in_terminal_async
from kani.engines.openai import OpenAIEngine
from kani.mcp import tools_from_mcp_servers

engine = OpenAIEngine(model="gpt-5-nano")
my_mcp_servers = [
    # ===== stdio servers =====
    # https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
    StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
    ),
    # https://github.com/ChromeDevTools/chrome-devtools-mcp
    StdioServerParameters(
        command="npx",
        args=["-y", "chrome-devtools-mcp@latest"],
    ),
    # ===== streamable HTTP server =====
    # https://github.com/cloudflare/mcp-server-cloudflare
    StreamableHttpParameters(
        url="https://docs.mcp.cloudflare.com/mcp",
    ),
]


async def run():
    async with tools_from_mcp_servers(my_mcp_servers) as mcp_tools:
        ai = Kani(engine, functions=mcp_tools)
        await chat_in_terminal_async(ai, show_function_args=True)


if __name__ == "__main__":
    asyncio.run(run())
