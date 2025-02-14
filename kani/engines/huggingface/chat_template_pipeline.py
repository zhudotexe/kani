"""
This module contains a fallback PromptPipeline intended to make kani as compatible with Hugging Face's Chat Templates
system (https://huggingface.co/docs/transformers/main/en/chat_templating) as possible, working around the limitations
that exist in that system. Specifically:

- Chat Templates are often written to enforce alternating user/assistant roles and raise errors if you don't
    - This makes measuring the message length of a single arbitrary message difficult
- Chat Templates often require at least one user message
    - This makes measuring the token length of function prompts difficult

Because of these limitations, we recommend creating a PromptPipeline for production implementations of
Hugging Face-hosted models. However, the default fallback implementation should be sufficient for development and
prototyping.
"""

import logging
import warnings
from collections import defaultdict
from functools import cached_property
from typing import Iterable

from kani import AIFunction, ChatMessage, ChatRole, PromptPipeline
from kani.exceptions import MissingModelDependencies
from kani.prompts.steps import ConversationDict

try:
    import torch
    import transformers
    from jinja2 import TemplateError
except ImportError:
    raise MissingModelDependencies(
        'The HuggingEngine requires extra dependencies. Please install kani with "pip install kani[huggingface]". '
        "You will also need to install PyTorch manually."
    ) from None


log = logging.getLogger(__name__)
OutputT = str | torch.Tensor


class ChatTemplatePromptPipeline(PromptPipeline[OutputT]):
    """
    Like a normal :class:`.PromptPipeline`, but the final step always applies the Hugging Face chat template
    associated with the given tokenizer.

    Use a ``.conversation_dict`` step to translate the roles of the messages, if needed.
    """

    def __init__(self, tokenizer, steps=None):
        super().__init__(steps)
        self.tokenizer = tokenizer
        self._ensure_chat_template()
        self._padding_len_by_role: dict[ChatRole, int] = defaultdict(lambda: 0)
        self._has_inferred_role_paddings = False

    @classmethod
    def from_pretrained(cls, model_id: str):
        """
        Create a ChatTemplatePromptPipeline from a model ID.
        Useful for applying a HF model's chat template to another engine.
        """
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        return cls(tok)

    # ===== auto reserve inference =====
    _chat_template_dummy_msg = {"role": "user", "content": "dummy"}

    @cached_property
    def _chat_template_dummy_len(self) -> int:
        return len(self.tokenizer.apply_chat_template([self._chat_template_dummy_msg], add_generation_prompt=False))

    def _infer_padding_len(self, role: ChatRole):
        """Set up _padding_len_by_role for each message type"""
        # try appending a dummy message to various bases, based on role
        for base_msgs in _padding_length_inference_base(role):
            try:
                log.debug(f"Estimating padding len for {role}...")
                msgs = base_msgs + [ChatMessage(role=role, content="dummy")]

                # get token len of base messages + dummy message
                conversation = super().execute(msgs)
                conversation_len = len(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False))

                # get token len of just base messages
                if base_msgs:
                    base_conversation = super().execute(base_msgs)
                    base_conversation_len = len(
                        self.tokenizer.apply_chat_template(base_conversation, add_generation_prompt=False)
                    )
                else:
                    base_conversation_len = 0

                # get token len of "dummy"
                text_len = len(self.tokenizer.encode("dummy", add_special_tokens=False))

                # padding = total - (base + dummy)
                self._padding_len_by_role[role] = max(conversation_len - (text_len + base_conversation_len), 0)
                log.debug(
                    f"{conversation_len=}, {base_conversation_len=}, {text_len=}, padding"
                    f" estimate={self._padding_len_by_role[role]}"
                )
            except (TemplateError, IndexError) as e:
                log.debug("Failed to estimate message padding length", exc_info=e)
                continue
            else:
                return
        # if we never found a base conversation that works, do a best guess
        log.warning(
            "Estimating message token padding with chat template application raised an error, assuming messages"
            " have a padding equal to length of role name plus 4 pad tokens. If this is incorrect, please implement"
            " a PromptPipeline.",
        )
        self._padding_len_by_role[role] = len(self.tokenizer.encode(role.value, add_special_tokens=False)) + 4

    def _chat_template_infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        # if at least one message is required, return a tensor with len equal to prompt w/ dummy minus dummy only
        full_len = len(self.tokenizer.apply_chat_template([self._chat_template_dummy_msg], add_generation_prompt=True))
        return torch.zeros((1, full_len - self._chat_template_dummy_len))  # the result gets cached in HuggingEngine

    # ===== pathological cases =====
    def _chat_template_message_len(self, message: ChatMessage) -> OutputT:
        """Estimate the message length of a single message based off the chat template."""
        conversation = super().execute([message])
        try:
            out_len = len(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False))
        except TemplateError:
            # the template probably enforces user/assistant,
            # return a best-effort estimate based on the estimated paddings for messages of this role
            raw_tok_len = len(self.tokenizer.encode(message.text, add_special_tokens=False))
            out_len = raw_tok_len + self._padding_len_by_role[message.role]
        return torch.zeros((1, out_len))

    def _chat_template_function_token_reserve(self, functions: list[AIFunction]) -> OutputT:
        """Estimate the function token reserve based off the chat template."""
        tools = [f.json_schema for f in functions] if functions else None
        full_len = len(
            self.tokenizer.apply_chat_template(
                [self._chat_template_dummy_msg], tools=tools, add_generation_prompt=False
            )
        )
        return torch.zeros((1, full_len - self._chat_template_dummy_len))

    # ===== prompt normal case =====
    def _chat_template_build_prompt(
        self, conversation: list[dict], functions: list[AIFunction] | None = None
    ) -> OutputT:
        """Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor.

        The default implementation uses the model tokenizer's `apply_chat_template` method.
        """
        tools = [f.json_schema for f in functions] if functions else None
        return self.tokenizer.apply_chat_template(conversation, tools=tools, add_generation_prompt=True, tokenize=False)

    # ===== utils =====
    def _ensure_chat_template(self):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise MissingModelDependencies(
                "To use the HuggingEngine with built-in chat templates requires `transformers>=4.34.0`. You currently"
                f" have `transformers=={transformers.__version__}`. Please update your transformers with `pip install"
                " -U transformers` or supply a `prompt_template` to this HuggingEngine."
            )

    # ===== overrides =====
    def execute(
        self, msgs: list[ChatMessage], functions: list[AIFunction] = None, *, deepcopy=False, for_measurement=False
    ) -> OutputT:
        if functions is None:
            functions = []
        if not any(isinstance(step, ConversationDict) for step in self.steps):
            debug_msg = (
                "ChatTemplatePromptPipeline expects the final output of the pipeline to be a list[dict] but no"
                " ConversationDict step was found in the pipeline, appending a default ConversationDict() step..."
            )
            # if the user defined steps we probably want to make this more visible; otherwise default behaviour is fine
            if self.steps:
                warnings.warn(debug_msg)
            else:
                log.debug(debug_msg)
            self.conversation_dict(additional_keys=hf_tool_use_keys)

        conversation = super().execute(msgs, functions, deepcopy=deepcopy, for_measurement=for_measurement)

        # infer role paddings if we have not yet done so
        if not self._has_inferred_role_paddings:
            self._has_inferred_role_paddings = True
            for role in ChatRole:
                self._infer_padding_len(role)

        # apply the chat template
        try:
            return self._chat_template_build_prompt(conversation, functions=functions)
        except (TemplateError, IndexError):
            # try and recover for these specific pathological cases
            if for_measurement:
                # one message - probably message len
                if len(msgs) == 1:
                    return self._chat_template_message_len(msgs[0])
                # empty message list and no functions, for measurement - probably token reserve
                if not (msgs or functions):
                    return self._chat_template_infer_token_reserve()
                # empty messages and functions - probably function reserve
                if not msgs:
                    return self._chat_template_function_token_reserve(functions)
            # uh oh
            raise

    def explain(self, *args, **kwargs):
        super().explain(*args, **kwargs)
        # print out inferred padding stats
        print(
            "\n### ChatTemplatePromptPipeline Stats\n*These metrics are inferred from the chat template and may be"
            " slightly off from the true count.*"
        )
        print(f"Token Reserve: {len(self._chat_template_infer_token_reserve()[0])}")
        print(f"Function Token Reserve: {len(self._chat_template_function_token_reserve([])[0])}")
        for role in ChatRole:
            print(f"{role.value} Role Padding: {self._padding_len_by_role[role]}")


def hf_tool_use_keys(message: ChatMessage) -> dict:
    """
    Given an ASSISTANT or FUNCTION message, return extra keys for tool calling models.

    See https://huggingface.co/docs/transformers/main/chat_templating#advanced-tool-use--function-calling.
    """
    if message.role == ChatRole.ASSISTANT and message.tool_calls:
        # ref:
        # tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
        # messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
        tcs = []
        for tc in message.tool_calls:
            tcs.append({"type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.kwargs}})
        return {"tool_calls": tcs}
    elif message.role == ChatRole.FUNCTION:
        # ref:
        # {"role": "tool", "name": "get_current_temperature", "content": "22.0", "tool_call_id": tool_call_id}
        data = {}
        if message.name is not None:
            data["name"] = message.name
        if message.tool_call_id is not None:
            data["tool_call_id"] = message.tool_call_id
        return data
    return {}


def _padding_length_inference_base(role: ChatRole) -> Iterable[list[ChatMessage]]:
    """Given a role, yield possible base messages used to infer padding."""
    base_with_system = [ChatMessage.system("dummy"), ChatMessage.user("dummy"), ChatMessage.assistant("dummy")]
    base_without_system = [ChatMessage.user("dummy"), ChatMessage.assistant("dummy")]
    if role == ChatRole.USER:
        yield base_with_system
        yield base_without_system
    elif role == ChatRole.ASSISTANT:
        yield base_with_system + [ChatMessage.user("dummy")]
        yield base_without_system + [ChatMessage.user("dummy")]
        yield [ChatMessage.system("dummy"), ChatMessage.user("dummy")]
        yield [ChatMessage.user("dummy")]
    elif role == ChatRole.FUNCTION:
        yield base_with_system
        yield base_without_system
    # SYSTEM: only empty
    # default, no conversation
    yield []
