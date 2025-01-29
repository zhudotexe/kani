import json
import logging
import re

from kani.engines.base import BaseCompletion
from kani.models import FunctionCall, ToolCall
from .base import BaseToolCallParser

log = logging.getLogger(__name__)


class MistralToolCallParser(BaseToolCallParser):
    """
    Tool calling adapter for Mistral models using the v3 or v7 tokenizer::

        --- v3 ---
        mistral-tiny-2407
        open-mixtral-8x22b-2404
        mistral-small-2409
        mistral-large-2407
        codestral-2405
        codestral-mamba-2407
        --- v7 ---
        mistral-large-2411
    """

    def __init__(self, *args, tool_call_start_token: str = "[TOOL_CALLS]", tool_call_end_token: str = "</s>", **kwargs):
        super().__init__(
            *args, tool_call_start_token=tool_call_start_token, tool_call_end_token=tool_call_end_token, **kwargs
        )

    def parse_tool_calls(self, content: str):
        tool_json = re.search(
            rf"{re.escape(self.tool_call_start_token)}\s*(.+?)\s*({re.escape(self.tool_call_end_token)})?$",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if tool_json is None:
            return content, []
        log.debug(f"Found tool JSON while parsing: {tool_json.group(1)}")
        actions = json.loads(tool_json.group(1))

        # translate back to kani spec
        tool_calls = []
        for action in actions:
            tool_name = action["name"]
            tool_args = json.dumps(action["arguments"])
            tool_id = action.get("id")
            tool_call = ToolCall.from_function_call(FunctionCall(name=tool_name, arguments=tool_args), call_id_=tool_id)
            tool_calls.append(tool_call)

        # return trimmed content and tool calls
        return content[: tool_json.start()], tool_calls

    async def predict(self, messages, functions=None, **hyperparams):
        hyperparams.setdefault("decode_kwargs", dict(skip_special_tokens=False))
        completion = await super().predict(messages, functions, **hyperparams)
        completion.message.content = completion.message.content.removesuffix(self.tool_call_end_token).strip()
        return completion

    async def stream(self, messages, functions=None, **hyperparams):
        hyperparams.setdefault("decode_kwargs", dict(skip_special_tokens=False))

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            if isinstance(elem, BaseCompletion):
                elem.message.content = elem.message.content.removesuffix(self.tool_call_end_token).strip()
                yield elem
            else:
                yield elem.removesuffix(self.tool_call_end_token)
