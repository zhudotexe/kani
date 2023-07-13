import asyncio
import inspect
import typing
from typing import AsyncIterable, Callable

import cachetools
import tiktoken  # todo openai
from pydantic import validate_call

from .aiparam import get_aiparam
from .engines.openai.client import OpenAIClient
from .exceptions import NoSuchFunction, WrappedCallException, FunctionCallException, FunctionSpecError
from .json_schema import AIParamSchema, create_json_schema
from .models import ChatMessage, FunctionSpec, FunctionCall, ChatRole


class Chatterbox:
    def __init__(
        self,
        client: OpenAIClient,
        system_prompt: str = None,
        always_include_messages: list[ChatMessage] = None,
        model="gpt-4",
        desired_response_tokens: int = 450,  # roughly the size of a discord message
        max_context_size: int = 8192,  # depends on model,
        chat_history: list[ChatMessage] = None,
        **hyperparams,
    ):
        self.client = client
        self.tokenizer = None
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self.model = model
        self.desired_response_tokens = desired_response_tokens
        self.max_context_size = max_context_size
        self.always_include_messages = ([ChatMessage.system(self.system_prompt)] if system_prompt else []) + (
            always_include_messages or []
        )
        self.chat_history: list[ChatMessage] = chat_history or []
        self.hyperparams = hyperparams

        # async to prevent generating multiple responses missing context
        self.lock = asyncio.Lock()

        # cache
        self._oldest_idx = 0
        self._reserve_tokens = 0  # something is adding N tokens to our prompt, e.g. func calling
        self._message_tokens = cachetools.FIFOCache(256)
        # todo move
        self._load_tokenizer()

    def _load_tokenizer(self):
        """
        Load the tokenizer (from the internet if first run). See
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def message_token_len(self, message: ChatMessage):
        """Returns the number of tokens used by a list of messages."""
        try:
            return self._message_tokens[message]
        except KeyError:
            mlen = len(self.tokenizer.encode(message.content)) + 5  # ChatML = 4, role = 1
            if message.name:
                mlen += len(self.tokenizer.encode(message.name))
            if message.function_call:
                mlen += len(self.tokenizer.encode(message.function_call.name))
                mlen += len(self.tokenizer.encode(message.function_call.arguments))
            self._message_tokens[message] = mlen
            return mlen

    async def get_truncated_chat_history(self):
        """
        Returns a list of messages such that the total token count in the messages is less than
        (4096 - desired_response_tokens).
        Always includes the system prompt plus any always_include_messages.
        """
        reversed_history = []
        always_len = sum(self.message_token_len(m) for m in self.always_include_messages) + self._reserve_tokens
        remaining = self.max_context_size - (always_len + self.desired_response_tokens)
        for idx in range(len(self.chat_history) - 1, self._oldest_idx - 1, -1):
            message = self.chat_history[idx]
            message_len = self.message_token_len(message)
            remaining -= message_len
            if remaining > 0:
                reversed_history.append(message)
            else:
                self._oldest_idx = idx + 1
                break
        return self.always_include_messages + reversed_history[::-1]

    # === main entrypoints ===
    async def load_tokenizer(self):
        await asyncio.get_event_loop().run_in_executor(None, self._load_tokenizer)

    async def chat_round(self, query: str, **kwargs) -> str:
        """Perform a single chat round (user -> model -> user, no functions allowed)."""
        async with self.lock:
            # get the user's chat input
            self.chat_history.append(ChatMessage.user(query.strip()))

            # get the context
            messages = await self.get_truncated_chat_history()

            # get the model's output, save it to chat history
            completion = await self.client.create_chat_completion(
                model=self.model, messages=messages, **self.hyperparams, **kwargs
            )

            message = completion.message
            self._message_tokens[message] = completion.usage.completion_tokens + 5
            self.chat_history.append(message)
            return message.content


class ChatterboxWithFunctions(Chatterbox):
    """Base class for a chatterbox with functions.

    Subclass and use ``@ai_function()`` to register functions. The schema will be autogenerated from the function
    signature (see :func:`ai_function`).

    To perform a chat round with functions, use ``full_round()`` as an async iterator::

        async for msg in chatterbox.full_round(prompt):
            # responses...

    Each response will be a str; you can control the format of a yielded function call with ``function_call_formatter``.

    **Retry & Model Feedback**
    If the model makes an error when attempting to call a function (e.g. calling a function that does not exist or
    passing params with incorrect and non-coercible types) or the function raises an exception, Chatterbox will send the
    error in a system message to the model, allowing it up to *retry_attempts* to correct itself and retry the call.
    """

    def __init__(self, *args, retry_attempts: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_attempts = retry_attempts

        # find all registered ai_functions and save them
        self.functions = {}
        self.functions_spec = []
        for name, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(member, "__ai_function__", None):
                continue
            inner = validate_call(member)
            f: AIFunction = AIFunction(inner, **member.__ai_function__)
            if f.name in self.functions:
                raise ValueError(f"FunctionSpec {f.name!r} is already registered!")
            self.functions[f.name] = f
            self.functions_spec.append(FunctionSpec(name=f.name, description=f.desc, parameters=f.json_schema))

    async def full_round(
        self,
        query: str,
        function_call_formatter: Callable[[ChatMessage], str | None] = lambda _: None,
        **kwargs,
    ) -> AsyncIterable[str]:
        retry = 0
        is_model_turn = True
        async with self.lock:
            self.chat_history.append(ChatMessage.user(query.strip()))

            while is_model_turn:
                is_model_turn = False

                # do the model prediction
                messages = await self.get_truncated_chat_history()
                completion = await self.client.create_chat_completion(
                    model=self.model, messages=messages, functions=self.functions_spec, **self.hyperparams, **kwargs
                )
                # calculate function calling reserve tokens on first run
                if self._reserve_tokens == 0 and self.functions:
                    self._reserve_tokens = max(
                        completion.usage.prompt_tokens - sum(self.message_token_len(m) for m in messages), 0
                    )
                # bookkeeping
                message = completion.message
                self._message_tokens[message] = completion.usage.completion_tokens + 5
                self.chat_history.append(message)
                if text := message.content:
                    yield text

                # if function call, do it and attempt retry if it's wrong
                # todo trim the parts that are wrong?
                if not message.function_call:
                    return

                if fn_msg := function_call_formatter(message):
                    yield fn_msg

                try:
                    is_model_turn = await self._do_function_call(message.function_call)
                except FunctionCallException as e:
                    # tell the model what went wrong
                    if isinstance(e, NoSuchFunction):
                        self.chat_history.append(
                            ChatMessage.system(
                                f"The function {e.name!r} is not defined. Only use the provided functions."
                            )
                        )
                    else:
                        self.chat_history.append(ChatMessage.function(message.function_call.name, str(e)))
                    # retry if we have retry attempts left
                    retry += 1
                    if retry > self.retry_attempts or not e.retry:
                        # disable function calling on the next go
                        kwargs = {**kwargs, "function_call": "none"}
                    continue
                else:
                    retry = 0

    async def _do_function_call(self, call: FunctionCall) -> bool:
        """Resolve a single function call.

        :returns: True (default) if the model should immediately react; False if the user speaks next.
        """
        # get func
        f = self.functions.get(call.name)
        if not f:
            raise NoSuchFunction(call.name)
        # call it
        try:
            result = await f(**call.kwargs)
        except Exception as e:
            raise WrappedCallException(f.auto_retry, e) from e
        # save the result to the chat history
        self.chat_history.append(ChatMessage.function(f.name, result))
        # yield whose turn it is
        return f.after == ChatRole.ASSISTANT


class AIFunction:
    def __init__(self, inner, after: ChatRole, name: str, desc: str, auto_retry: bool, json_schema: dict = None):
        self.inner = inner
        self.after = after
        self.name = name
        self.desc = desc
        self.auto_retry = auto_retry
        self.json_schema = self.create_json_schema() if json_schema is None else json_schema

        # wraps() things
        self.__name__ = inner.__name__
        self.__qualname__ = inner.__qualname__
        self.__annotations__ = inner.__annotations__
        self.__module__ = inner.__module__
        self.__doc__ = inner.__doc__

    async def __call__(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result

    def create_json_schema(self) -> dict:
        """create a JSON schema representing this function's parameters as a JSON object."""
        # get list of params
        params = []
        sig = inspect.signature(self.inner)
        type_hints = typing.get_type_hints(self.inner)
        for name, param in sig.parameters.items():
            # ensure param can be supplied thru kwargs
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise FunctionSpecError(
                    "Positional-only or variadic parameters are not allowed in @ai_function()s."
                    f" ({self.inner.__name__}#{name})"
                )

            # ensure the type is annotated
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise FunctionSpecError(
                    f"All @ai_function() parameters must have a type annotation ({self.inner.__name__}#{name})."
                )

            # ensure type hint matches up
            if name not in type_hints:
                raise RuntimeError(f"The schema generator could not find the type hint ({self.inner.__name__}#{name}).")

            # get aiparam and add it to the list
            ai_param = get_aiparam(annotation)
            params.append(AIParamSchema(name=name, t=type_hints[name], default=param.default, aiparam=ai_param))
        # create a schema generator and generate
        return create_json_schema(params)


def ai_function(
    func=None,
    *,
    after: ChatRole = ChatRole.ASSISTANT,
    name: str = None,
    desc: str = None,
    auto_retry: bool = True,
    json_schema: dict | None = None,
):
    """Decorator to mark a method of a ChatterboxWithFunctions to expose to the AI.

    **Type Annotations**
    Chatterbox will automatically generate the function schema based off of the function's type annotations.
    The allowed types are:

    - Python primitive types (``None``, :class:`bool`, :class:`str`, :class:`int`, :class:`float`)
    - an enum (subclass of ``enum.Enum``)
    - a list or dict of the above types (e.g. ``list[str]``, ``dict[str, int]``, ``list[SomeEnum]``)

    When the AI calls into the function, Chatterbox guarantees that the passed parameters are of the annotated type.

    **Name & Descriptions**
    If not specified, the function description will be taken from its docstring, and name from the source.
    To specify descriptions of or override the name of a parameter, provide an :class:`AIParam` annotation using an
    Annotated type annotation.

    **Next Actor**
    After a function call returns, Chatterbox will hand control back to the LM to generate a response. If instead
    control should be given to the human (i.e. return from the chat round), set ``after=ChatRole.USER``.

    **Example**
    Here is an example of how you might implement a function to get weather::

        class Unit(enum.Enum):
            FAHRENHEIT = "fahrenheit"
            CELSIUS = "celsius"

        @ai_function()
        async def get_weather(
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            unit: Unit,
        ):
            \"""Get the current weather in a given location.\"""
            ...

    :param after: After completing the function call, who should speak next.
    :param name: The name of the function (defaults to f.__name__)
    :param desc: The desc of the function (defaults to docstring)
    :param auto_retry: Whether the model should retry calling the function if it gets it wrong.
    :param json_schema: If not using autogeneration, the JSON Schema to provide the model.
    """

    def deco(f):
        f.__ai_function__ = {
            "after": after,
            "name": name or f.__name__,
            "desc": desc or inspect.getdoc(f),
            "auto_retry": auto_retry,
            "json_schema": json_schema,
        }
        return f

    if func is not None:
        return deco(func)
    return deco
