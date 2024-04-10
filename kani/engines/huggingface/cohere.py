import functools
import json
import re
from collections.abc import AsyncIterable
from threading import Thread

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
from kani.prompts.impl.cohere import (
    COMMAND_R_PIPELINE,
    DEFAULT_PREAMBLE,
    DEFAULT_TASK,
    DEFAULT_TOOL_INSTRUCTIONS,
    DEFAULT_TOOL_PROMPT,
    build_rag_pipeline,
    build_tool_pipeline,
    function_prompt,
    tool_call_formatter,
)
from .base import HuggingEngine
from ..base import Completion

try:
    import torch
    from torch import tensor
    from transformers import TextIteratorStreamer
except ImportError:
    raise MissingModelDependencies(
        'The CommandREngine requires extra dependencies. Please install kani with "pip install'
        " 'kani[huggingface]'\". You will also need to install PyTorch manually."
    ) from None


class CommandREngine(HuggingEngine):
    """Implementation of Command R (35B) and Command R+ (104B) using huggingface transformers.

    Model IDs:

    - ``CohereForAI/c4ai-command-r-v01``
    - ``CohereForAI/c4ai-command-r-plus``

    **GPU Support**

    By default, the CommandREngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    **Usage**

    .. code-block:: python

        engine = CommandREngine("CohereForAI/c4ai-command-r-v01")
        ai = KaniWithFunctions(engine)

    **Configuration**

    Command R has many configurations that enable function calling and/or RAG, and it is poorly documented exactly
    how certain prompts affect the model. In this implementation, we default to the Cohere-supplied "preamble" if
    function definitions are supplied, and assume that we pass every generated function call and results each turn.

    When generating the result of a tool call turn, this implementation does NOT request the model to generate
    citations by default (unlike the Cohere API). You can enable citations by setting the ``rag_prompt_instructions``
    parameter to ``DEFAULT_RAG_INSTRUCTIONS_ACC`` or ``DEFAULT_RAG_INSTRUCTIONS_FAST`` (imported from
    ``kani.prompts.impl.cohere``).

    See the constructor's available parameters for more information.

    .. caution::

        Command R requires ``transformers>=4.39.1`` as a dependency. If you see warnings about a missing
        ``CohereTokenizerFast``, please update your version with ``pip install transformers>=4.39.1``.

    .. seealso:: https://huggingface.co/CohereForAI/c4ai-command-r-v01

    .. tip:: See :ref:`4b_quant` for information about loading a quantized model for lower memory usage.
    """

    token_reserve = 200  # generous reserve due to large ctx size and weird 3-mode prompt

    def __init__(
        self,
        model_id: str = "CohereForAI/c4ai-command-r-v01",
        *args,
        tool_prompt_include_function_calls=True,
        tool_prompt_include_function_results=True,
        tool_prompt_instructions=DEFAULT_TOOL_INSTRUCTIONS,
        rag_prompt_include_function_results=True,
        rag_prompt_instructions=None,
        **kwargs,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model (defaults to Command R's size of 128k).
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param tool_prompt_include_function_calls: Whether to include previous turns' function calls or just the model's
            answers.
        :param tool_prompt_include_function_results: Whether to include the results of previous turns' function calls in
            the context.
        :param tool_prompt_instructions: The system prompt to send just before the model's generation turn that includes
            instructions on the format to generate tool calls in. Generally you shouldn't change this.
        :param rag_prompt_include_function_results: Whether to include the results of previous turns' function calls in
            the context.
        :param rag_prompt_instructions: The system prompt to send just before the model's generation turn that includes
            instructions on the format to generate the result in. Can be None to only generate a model turn. Defaults
            to ``None`` to for maximum interoperability between models. Options:

            - ``from kani.prompts.impl.cohere import DEFAULT_RAG_INSTRUCTIONS_ACC``
            - ``from kani.prompts.impl.cohere import DEFAULT_RAG_INSTRUCTIONS_FAST``
            - ``None`` (default)
            - another user-supplied string
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        kwargs.setdefault("max_context_size", 128000)
        super().__init__(model_id, *args, **kwargs)

        self._tool_prompt_include_function_calls = tool_prompt_include_function_calls

        self._tool_pipeline = build_tool_pipeline(
            include_function_calls=tool_prompt_include_function_calls,
            include_function_results=tool_prompt_include_function_results,
            tool_instructions=tool_prompt_instructions,
        )
        self._rag_pipeline = build_rag_pipeline(
            include_previous_results=rag_prompt_include_function_results,
            rag_instructions=rag_prompt_instructions,
        )

    # ==== token counting ====
    def message_len(self, message: ChatMessage) -> int:
        # prompt str to tokens
        if message.text:
            tokenized = self.tokenizer.encode(message.text, add_special_tokens=False)
        else:
            tokenized = 0

        # worst-case function calls if we have them
        if self._tool_prompt_include_function_calls and message.role == ChatRole.ASSISTANT:
            func_body = tool_call_formatter(message)
            tokenized = self.tokenizer.encode(func_body, add_special_tokens=False)
            return len(tokenized) + 3

        # <results></results>
        if message.role == ChatRole.FUNCTION:
            return len(tokenized) + 12

        return len(tokenized) + 3

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        # include the additional default system prompt tokens here plus the directly_answer tool
        default_prompt_tokens = 325
        function_text = "\n\n".join(map(function_prompt, functions))
        function_tokens = len(self.tokenizer.encode(function_text, add_special_tokens=False))
        return function_tokens + default_prompt_tokens

    # ==== prompt ====
    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | torch.Tensor:
        # no functions: we can just do the default simple format
        if not functions:
            return COMMAND_R_PIPELINE(messages)

        # if we do have functions things get wacky
        # is the last message a FUNCTION? if so, we need to use the RAG template
        if messages and messages[-1].role == ChatRole.FUNCTION:
            return self._build_prompt_rag(messages)

        # otherwise use the TOOL template
        return self._build_prompt_tools(messages, functions)

    def _build_prompt_tools(self, messages: list[ChatMessage], functions: list[AIFunction]):
        # get the function definitions
        function_text = "\n\n".join(map(function_prompt, functions))
        tool_prompt = DEFAULT_TOOL_PROMPT.format(user_functions=function_text)

        # wrap the initial system message, if any
        if messages and messages[0].role == ChatRole.SYSTEM:
            messages[0] = messages[0].copy_with(content=DEFAULT_PREAMBLE + messages[0].text + tool_prompt)
        # otherwise add it in
        else:
            messages.insert(0, ChatMessage.system(DEFAULT_PREAMBLE + DEFAULT_TASK + tool_prompt))

        return self._tool_pipeline(messages)

    def _build_prompt_rag(self, messages: list[ChatMessage]):
        # wrap the initial system message, if any
        if messages and messages[0].role == ChatRole.SYSTEM:
            messages[0] = messages[0].copy_with(content=DEFAULT_PREAMBLE + messages[0].text)
        # otherwise add it in
        else:
            messages.insert(0, ChatMessage.system(DEFAULT_PREAMBLE + DEFAULT_TASK))

        return self._rag_pipeline(messages)

    # ==== generate ====
    def _generate(self, input_toks, input_len, hyperparams, functions):
        """Generate and return a completion (may be a directly_answer call)."""
        # run it through the model
        output = self.model.generate(input_toks, **hyperparams)
        # decode to tokens
        # the completion shouldn't include the prompt or stop token
        content = self.tokenizer.decode(output[0][input_len:-1]).strip()
        completion_tokens = len(output[0]) - (input_len + 1)

        # if we have tools, possibly parse out the Action
        tool_calls = None
        if functions and (action_json := re.match(r"Action:\s*```json\n(.+)\n```", content, re.IGNORECASE)):
            actions = json.loads(action_json.group(1))

            # translate back to kani spec
            tool_calls = []
            for action in actions:
                tool_name = action["tool_name"]
                tool_args = json.dumps(action["parameters"])
                tool_call = ToolCall.from_function_call(FunctionCall(name=tool_name, arguments=tool_args))
                tool_calls.append(tool_call)

        return Completion(
            ChatMessage.assistant(content, tool_calls=tool_calls),
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
        )

    async def _stream(self, input_toks, hyperparams, *, streamer_timeout=None) -> AsyncIterable[str | Completion]:
        """Low-level stream yielder (kind of weird duplicated code but it's ok)"""
        # noinspection DuplicatedCode
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=streamer_timeout
        )

        # run it through the model in another thread so that we can get the tokens in this thread
        generate_func = functools.partial(self.model.generate, input_toks, streamer=streamer, **hyperparams)
        thread = Thread(target=generate_func)
        thread.start()

        # then wait for tokens from the task
        for token in streamer:
            yield token

        # finally clean up the thread
        thread.join()

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        prompt = self.build_prompt(messages, functions)
        input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
        completion = self._generate(input_toks, input_len, hyperparams, functions)

        tool_calls = completion.message.tool_calls or []
        # if the model generated multiple calls that happen to include a directly_answer, remove the directly_answer
        if len(tool_calls) > 1:
            completion.message.tool_calls = [
                tc for tc in completion.message.tool_calls if tc.function.name != "directly_answer"
            ]
        # if tool says directly answer, call again with the rag pipeline (but no result)
        elif len(tool_calls) == 1 and tool_calls[0].function.name == "directly_answer":
            prompt = self._build_prompt_rag(messages)
            input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
            completion = self._generate(input_toks, input_len, hyperparams, functions)
        # otherwise don't touch it
        return completion

    async def stream(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        streamer_timeout=None,
        **hyperparams,
    ) -> AsyncIterable[str | Completion]:
        # if we have functions things get weird
        # if we have tools and the last turn is not FUNCTION, no-stream the first round to get the Action
        if functions and not (messages and messages[-1].role == ChatRole.FUNCTION):
            prompt = self.build_prompt(messages, functions)
            input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
            completion = self._generate(input_toks, input_len, hyperparams, functions)

            tool_calls = completion.message.tool_calls or []
            # if the model generated multiple calls that happen to include a directly_answer, remove the directly_answer
            if len(tool_calls) > 1:
                completion.message.tool_calls = [
                    tc for tc in completion.message.tool_calls if tc.function.name != "directly_answer"
                ]
            # if tool says directly answer, stream with the rag pipeline (but no result)
            elif len(tool_calls) == 1 and tool_calls[0].function.name == "directly_answer":
                prompt = self._build_prompt_rag(messages)
                input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
                async for elem in self._stream(input_toks, hyperparams, streamer_timeout=streamer_timeout):
                    yield elem
            # otherwise yield as normal
            else:
                yield completion
        # otherwise stream as normal
        else:
            async for elem in super().stream(messages, functions, streamer_timeout=streamer_timeout, **hyperparams):
                yield elem
