import base64
import contextlib
import functools
import io
import mimetypes

from kani import AIFunction, _optional

try:
    from mcp.client.session_group import ClientSessionGroup, ServerParameters
    from mcp.types import AudioContent, ImageContent, TextContent
except ImportError as e:
    raise ImportError(
        'Using Kani with MCP requires extra dependencies. Please install kani with "pip install kani[mcp]".'
    ) from None


@contextlib.asynccontextmanager
async def tools_from_mcp_servers(
    mcp_servers: list[ServerParameters],
    allowed_tools: list[str] = None,
    blocked_tools: list[str] = None,
    *,
    component_name_hook=None,
):
    """
    An async context manager to retrieve tools from MCP servers and open a connection in order to call these tools.
    Returns a list of Kani AIFunctions for each MCP tool in the specified server list.

    Usage:

    .. code-block:: python

        from kani import Kani
        from kani.engines.openai import OpenAIEngine
        from kani.mcp import tools_from_mcp_servers
        from mcp.client.session_group import StdioServerParameters, StreamableHttpParameters

        engine = OpenAIEngine("gpt-5-nano")
        my_mcp_servers = [
            # stdio server
            StdioServerParameters(
                command="python",
                args=["path_to_your_working_directory/mcp_server.py"],
            ),
            # streamable HTTP server
            StreamableHttpParameters(
                url="https://example.com/my-mcp-server",
                headers={"Authorization": "Bearer my-api-key"},
            )
        ]

        async def run():
            async with tools_from_mcp_servers(my_mcp_servers) as mcp_tools:
                ai = Kani(engine, functions=mcp_tools)
                async for msg in ai.full_round("Roll me damage for a 3rd level Fireball"):
                    print(msg)

        asyncio.run(run())

    :param mcp_servers: A list of MCP server configurations. Each server's tools will be exposed as AIFunctions
            to this Kani.
    :param allowed_tools: A list of MCP tool names to expose. Defaults to allowing all tools from each server.
    :param blocked_tools: A list of MCP tool names to block. At most one of ``allowed_tools`` or ``blocked_tools`` may
        be specified.
    :param component_name_hook: A callable function consuming (component_name, serverInfo) for custom names.
        This is provide a means to mitigate naming conflicts across servers.
        Example: ``lambda tool_name, serverInfo: f"{serverInfo.name}.{tool_name}"``
    """
    if allowed_tools is not None and blocked_tools is not None:
        raise ValueError("At most one of allowed_tools or blocked_tools may be specified.")

    # create a sessiongroup for the list of specified tools
    async with ClientSessionGroup(component_name_hook=component_name_hook) as grp:
        for server in mcp_servers:
            await grp.connect_to_server(server)

        # create AIFunctions for each tool
        aifunctions = []
        for name, tool in grp.tools.items():
            if allowed_tools is not None and name not in allowed_tools:
                continue
            if blocked_tools is not None and name in blocked_tools:
                continue

            async def _call_mcp_tool(__name, /, **kwargs):
                result = await grp.call_tool(__name, arguments=kwargs)
                # the tool returns a list of content blocks
                if len(result.content) == 1 and result.content[0].type == "text":
                    return result.content[0].text

                out = []
                # TextContent | ImageContent | AudioContent
                for block in result.content:
                    if isinstance(block, TextContent):
                        out.append(block.text)
                    elif isinstance(block, ImageContent):
                        extensions = [
                            e.removeprefix(".") for e in mimetypes.guess_all_extensions(block.mimeType, strict=False)
                        ]
                        out.append(_optional.multimodal_core.ImagePart.from_b64(block.data, formats=extensions or None))
                    elif isinstance(block, AudioContent):
                        f = io.BytesIO(base64.b64decode(block.data))
                        f.seek(0)
                        out.append(_optional.multimodal_core.AudioPart.from_file(f))
                    else:
                        raise ValueError(f"Unsupported MCP return value: {block}")
                return out

            aifunctions.append(
                AIFunction(
                    functools.partial(_call_mcp_tool, name),
                    name=name,
                    desc=tool.description,
                    json_schema=tool.inputSchema,
                )
            )
        yield aifunctions
