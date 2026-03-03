"""Helpers to translate kani chat objects into OpenAI params for the Responses API."""

import uuid

from kani import _optional
from kani.engines.base import Completion
from kani.engines.openai.parts import OpenAIReasoningPart, OpenAIUnknownPart
from kani.engines.openai.utils import DottableDict
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole, FunctionCall, MessagePart, ToolCall
from kani.parts import ReasoningPart
from kani.utils.warnings import warn_in_userspace

try:
    from openai.types.responses.response_input_item_param import FunctionCallOutput
    from openai.types.responses import (
        EasyInputMessageParam,
        Response,
        ResponseFunctionCallOutputItemParam,
        ResponseFunctionToolCall,
        ResponseFunctionToolCallParam,
        ResponseInputContentParam,
        ResponseInputFileContentParam,
        ResponseInputImageContentParam,
        ResponseInputImageParam,
        ResponseInputItemParam,
        ResponseInputTextContentParam,
        ResponseInputTextParam,
        ResponseOutputItem,
        ResponseOutputMessage,
        ResponseOutputRefusal,
        ResponseOutputText,
        ResponseReasoningItem,
        ResponseReasoningItemParam,
    )
    from openai.types.shared_params import FunctionDefinition
except ImportError as e:
    raise MissingModelDependencies(
        'The OpenAIEngine requires extra dependencies. Please install kani with "pip install kani[openai]".'
    ) from None

OAI_RESPONSES_EXTRA_KEY = "openai_response"


# ==== kani -> openai ====
# decomp
def kani_cm_to_openai_responses_inputs(msg: ChatMessage) -> list[ResponseInputItemParam]:
    """Translate a kani ChatMessage into a list of ResponseInputItems."""
    # if we already have the translated content from google, just use that
    # this is useful for thought signatures, mainly
    if OAI_RESPONSES_EXTRA_KEY in msg.extra:
        return msg.extra[OAI_RESPONSES_EXTRA_KEY].output

    # translate tool responses to a function to the right openai format
    match msg.role:
        case ChatRole.FUNCTION if msg.tool_call_id is not None:
            return [
                FunctionCallOutput(
                    type="function_call_output", output=_parts_to_oai_responses(msg.parts), call_id=msg.tool_call_id
                )
            ]
        case ChatRole.FUNCTION:
            # not allowed in the responses API
            raise ValueError("Unbound function message found! This should not happen.")
        case ChatRole.SYSTEM | ChatRole.USER:
            return [_kani_cm_to_responses_message(msg)]
        case _:  # assistant
            asst_parts: list[ResponseInputItemParam] = []

            # reasoning
            for part in msg.parts:
                if isinstance(part, OpenAIReasoningPart):
                    asst_parts.append(
                        ResponseReasoningItemParam(
                            type="reasoning",
                            id=part.id,
                            summary=[{"type": "summary_text", "text": part.content}],
                            encrypted_content=part.encrypted_content,
                        )
                    )
                elif isinstance(part, ReasoningPart):
                    # this should only really happen when manually defining prompts, since we'd have otherwise
                    # copied it from the extra, so ganbarimashou
                    asst_parts.append(
                        ResponseReasoningItemParam(
                            type="reasoning",
                            id=part.extra.get("_openai_id", str(uuid.uuid4())),
                            summary=[{"type": "summary_text", "text": part.content}],
                            content=[{"type": "reasoning_text", "text": part.content}],
                            encrypted_content=part.extra.get("_openai_encrypted_content"),
                        )
                    )
                elif isinstance(part, OpenAIUnknownPart):
                    asst_parts.append(part.data)

            # main content
            asst_parts.append(_kani_cm_to_responses_message(msg))

            # tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    asst_parts.append(
                        ResponseFunctionToolCallParam(
                            type="function_call", call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments
                        )
                    )

            return asst_parts


def _kani_cm_to_responses_message(msg: ChatMessage) -> EasyInputMessageParam:
    match msg:
        case ChatMessage(content=list(parts)):
            content = _parts_to_oai_responses(parts)
        case _:
            content = msg.text
    return EasyInputMessageParam(role=msg.role.value, content=content)


# --- multimodal ---
def _parts_to_oai_responses(parts: list[MessagePart | str]) -> str | list[ResponseInputContentParam]:
    """Translate a list of Kani messageparts into openai message components."""
    if _optional.has_multimodal_core:
        out = []
        for part in parts:
            if isinstance(part, _optional.multimodal_core.ImagePart):
                data_uri = part.as_b64_uri()
                out.append(ResponseInputImageParam(type="input_image", image_url=data_uri, detail="auto"))
            elif isinstance(part, _optional.multimodal_core.BaseMultimodalPart):
                warn_in_userspace(
                    f"The OpenAI Responses API does not support the {type(part).__name__} multimodal input. Consider"
                    " using the Chat Completions API for audio."
                )
                out.append(ResponseInputTextParam(type="input_text", text=str(part)))
            else:
                out.append(ResponseInputTextParam(type="input_text", text=str(part)))
        return out
    # no multimodal base case
    return "".join(map(str, parts))


# ==== openai -> kani ====
def openai_responses_response_to_kani_completion(response: Response) -> Completion:
    msg = openai_responses_outputs_to_kani_cm(response.output)
    msg.extra[OAI_RESPONSES_EXTRA_KEY] = DottableDict(
        response.model_dump(
            mode="json",
            exclude_unset=True,
            exclude={
                "output": {
                    "__all__": {
                        "parsed_arguments": True,  # function calling
                        "content": {"__all__": {"parsed_content": True}},  # reasoning or annotations?
                    }
                }
            },  # streaming API returns extra keys which cause problems
        )
    )
    msg.extra["openai_usage"] = DottableDict(response.usage.model_dump(mode="json", exclude_unset=True))
    return Completion(
        message=msg, prompt_tokens=response.usage.input_tokens, completion_tokens=response.usage.output_tokens
    )


def openai_responses_outputs_to_kani_cm(outputs: list[ResponseOutputItem]) -> ChatMessage:
    """Translate a list of OpenAI responses outputs into a kani ChatMessage."""
    content = []
    tool_calls = []
    for item in outputs:
        match item:
            # collect reasoning
            case ResponseReasoningItem():
                content.append(
                    OpenAIReasoningPart(
                        id=item.id,
                        encrypted_content=item.encrypted_content,
                        content="\n".join(i.text for i in item.content or item.summary),
                    )
                )
            # collect tool calls
            case ResponseFunctionToolCall():
                tool_calls.append(
                    ToolCall(
                        id=item.call_id, type=item.type, function=FunctionCall(name=item.name, arguments=item.arguments)
                    )
                )
            # collect content
            case ResponseOutputMessage():
                for content_item in item.content:
                    match content_item:
                        case ResponseOutputText(text=text) | ResponseOutputRefusal(refusal=text):
                            content.append(text)
            # collect other unknown parts
            case _:
                content.append(OpenAIUnknownPart(type=item.type, data=item.model_dump(mode="json")))

    return ChatMessage(role=ChatRole.ASSISTANT, content=content, tool_calls=tool_calls or None)
