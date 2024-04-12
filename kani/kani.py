import asyncio
import inspect
import logging
import warnings
from collections.abc import AsyncIterable
from typing import Callable

from .ai_function import AIFunction
from .engines.base import BaseCompletion, BaseEngine
from .exceptions import FunctionCallException, MessageTooLong, NoSuchFunction, WrappedCallException
from .internal import ExceptionHandleResult, FunctionCallResult
from .models import ChatMessage, ChatRole, FunctionCall, QueryType, ToolCall
from .streaming import DummyStream, StreamManager
from .utils.message_formatters import assistant_message_contents
from .utils.typing import PathLike, SavedKani

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


class Kani:
    """Base class for all kani.

    **Entrypoints**

    ``chat_round(query: str, **kwargs) -> ChatMessage``

    ``chat_round_str(query: str, **kwargs) -> str``

    ``chat_round_stream(query: str, **kwargs) -> StreamManager``

    ``full_round(query: str, **kwargs) -> AsyncIterable[ChatMessage]``

    ``full_round_str(query: str, message_formatter: Callable[[ChatMessage], str], **kwargs) -> AsyncIterable[str]``

    ``full_round_stream(query: str, **kwargs) -> AsyncIterable[StreamManager]``

    **Function Calling**

    Subclass and use ``@ai_function()`` to register functions. The schema will be autogenerated from the function
    signature (see :func:`ai_function`).

    To perform a chat round with functions, use :meth:`full_round()` as an async iterator::

        async for msg in kani.full_round(prompt):
            # responses...

    Each response will be a :class:`.ChatMessage`.

    Alternatively, you can use :meth:`full_round_str` and control the format of a yielded function call with
    ``function_call_formatter``.

    **Retry & Model Feedback**

    If the model makes an error when attempting to call a function (e.g. calling a function that does not exist or
    passing params with incorrect and non-coercible types) or the function raises an exception, Kani will send the
    error in a system message to the model, allowing it up to *retry_attempts* to correct itself and retry the call.
    """

    def __init__(
        self,
        engine: BaseEngine,
        system_prompt: str = None,
        always_included_messages: list[ChatMessage] = None,
        desired_response_tokens: int = 450,
        chat_history: list[ChatMessage] = None,
        functions: list[AIFunction] = None,
        retry_attempts: int = 1,
    ):
        """
        :param engine: The LM engine implementation to use.
        :param system_prompt: The system prompt to provide to the LM. The prompt will not be included in
            :attr:`chat_history`.
        :param always_included_messages: A list of messages to always include as a prefix in all chat rounds (i.e.,
            evict newer messages rather than these to manage context length). These will not be included in
            :attr:`chat_history`.
        :param desired_response_tokens: The minimum amount of space to leave in ``max context size - tokens in prompt``.
            To control the maximum number of tokens generated more precisely, you may be able to configure the engine
            (e.g. ``OpenAIEngine(..., max_tokens=250)``).
        :param chat_history: The chat history to start with (not including system prompt or always included messages),
            for advanced use cases. By default, each kani starts with a new conversation session.

            .. caution::
                If you pass another kani's chat history here without copying it, the same list will be mutated!
                Use ``chat_history=mykani.chat_history.copy()`` to pass a copy.
        :param functions: A list of :class:`.AIFunction` to expose to the model (for dynamic function calling).
            Use :func:`.ai_function` to define static functions (see :doc:`function_calling`).
        :param retry_attempts: How many attempts the LM may take per full round if any tool call raises an exception.
        """
        self.engine = engine
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self.desired_response_tokens = desired_response_tokens
        self.max_context_size = engine.max_context_size

        self.always_included_messages: list[ChatMessage] = (
            [ChatMessage.system(self.system_prompt)] if system_prompt else []
        ) + (always_included_messages or [])
        """Chat messages that are always included as a prefix in the model's prompt.
        Includes the system message, if supplied."""

        self.chat_history: list[ChatMessage] = chat_history or []
        """All messages in the current chat state, not including system or always included messages."""

        # async to prevent generating multiple responses missing context
        self.lock = asyncio.Lock()

        # function calling
        self.retry_attempts = retry_attempts

        # find all registered ai_functions and save them
        if functions is None:
            functions = []
        self.functions: dict[str, AIFunction] = {f.name: f for f in functions}
        for name, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if not hasattr(member, "__ai_function__"):
                continue
            f: AIFunction = AIFunction(member, **member.__ai_function__)
            if f.name in self.functions:
                raise ValueError(f"AIFunction {f.name!r} is already registered!")
            self.functions[f.name] = f

    # ==== internals ====
    async def _chat_round_before(self, query: QueryType, **kwargs):
        """Common preflight for chat_round_*, returns the kwargs for get_model_completion/stream"""
        # warn if the user has functions defined and has not explicitly silenced them in this call
        if self.functions and "include_functions" not in kwargs:
            warnings.warn(
                f"You have defined functions in the body of {type(self).__name__} but chat_round() will not call"
                " functions. Use full_round() instead.\nIf this is intentional, use chat_round(...,"
                " include_functions=False) to silence this warning."
            )
        kwargs["include_functions"] = False
        # add the user's chat input to the state
        if query is not None:
            await self.add_to_history(ChatMessage.user(query))
        return kwargs

    async def _add_completion_to_history(self, completion: BaseCompletion):
        """Add the message in the completion to the chat history and return it"""
        message = completion.message
        await self.add_to_history(message)
        return message

    async def _full_round(self, query: QueryType, *, _kani_is_stream=False, **kwargs):
        """Underlying handler for full_round with stream support."""
        retry = 0
        is_model_turn = True
        async with self.lock:
            if query is not None:
                await self.add_to_history(ChatMessage.user(query))

            while is_model_turn:
                # do the model prediction (stream or no stream)
                if _kani_is_stream:
                    stream = self.get_model_stream(**kwargs)
                    manager = StreamManager(stream, role=ChatRole.ASSISTANT, after=self._add_completion_to_history)
                    yield manager
                    message = await manager.message()
                else:
                    completion = await self.get_model_completion(**kwargs)
                    message = await self._add_completion_to_history(completion)
                    yield message

                # if function call, do it and attempt retry if it's wrong
                if not message.tool_calls:
                    return

                # run each tool call in parallel
                async def _do_tool_call(tc: ToolCall):
                    # call the method and set the is_tool_call_error attr (if the impl has not already set it)
                    try:
                        tc_result = await self.do_function_call(tc.function, tool_call_id=tc.id)
                        if tc_result.message.is_tool_call_error is None:
                            tc_result.message.is_tool_call_error = False
                    except FunctionCallException as e:
                        tc_result = await self.handle_function_call_exception(tc.function, e, retry, tool_call_id=tc.id)
                        tc_result.message.is_tool_call_error = True
                    return tc_result

                # and update results after they are completed
                is_model_turn = False
                should_retry_call = False
                n_errs = 0
                results = await asyncio.gather(*(_do_tool_call(tc) for tc in message.tool_calls))
                for result in results:
                    # save the result to the chat history
                    await self.add_to_history(result.message)

                    # yield it, possibly in dummy streammanager
                    if _kani_is_stream:
                        yield DummyStream(result.message)
                    else:
                        yield result.message

                    if isinstance(result, ExceptionHandleResult):
                        is_model_turn = True
                        n_errs += 1
                        # retry if any function says so
                        should_retry_call = should_retry_call or result.should_retry
                    else:
                        # allow model to generate response if any function says so
                        is_model_turn = is_model_turn or result.is_model_turn

                # if we encountered an error, increment the retry counter and allow the model to generate a response
                if n_errs:
                    retry += 1
                    if not should_retry_call:
                        # disable function calling on the next go
                        kwargs["include_functions"] = False
                else:
                    retry = 0

    # === main entrypoints ===
    async def chat_round(self, query: QueryType, **kwargs) -> ChatMessage:
        """Perform a single chat round (user -> model -> user, no functions allowed).

        :param query: The contents of the user's chat message. Can be None to generate a completion without a user
            prompt.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        :returns: The model's reply.
        """
        async with self.lock:
            kwargs = await self._chat_round_before(query, **kwargs)
            completion = await self.get_model_completion(**kwargs)
            return await self._add_completion_to_history(completion)

    async def chat_round_str(self, query: QueryType, **kwargs) -> str:
        """Like :meth:`chat_round`, but only returns the text content of the message."""
        msg = await self.chat_round(query, **kwargs)
        return msg.text

    def chat_round_stream(self, query: QueryType, **kwargs) -> StreamManager:
        """
        Returns a stream of tokens from the engine as they are generated.

        To consume tokens from a stream, use this class as so::

            stream = ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")
            async for token in stream:
                print(token, end="")
            msg = await stream.message()

        .. tip::
            For compatibility and ease of refactoring, awaiting the stream itself will also return the message, i.e.::

                msg = await ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")

            (note the ``await`` that is not present in the above examples).

        The arguments are the same as :meth:`chat_round`.
        """

        # this is kind of cursed - we need to run the preflight stuff before we start yielding tokens but
        # this is a synch context so we'll delegate the iterator to do it before it starts yielding
        async def _impl():
            _kwargs = await self._chat_round_before(query, **kwargs)
            async for elem in self.get_model_stream(**_kwargs):
                yield elem

        return StreamManager(_impl(), role=ChatRole.ASSISTANT, after=self._add_completion_to_history, lock=self.lock)

    async def full_round(self, query: QueryType, **kwargs) -> AsyncIterable[ChatMessage]:
        """Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields each non-user ChatMessage created during the round.
        A ChatMessage will have at least one of ``(content, function_call)``.

        Use this in an async for loop, like so::

            async for msg in kani.full_round("How's the weather?"):
                print(msg.text)

        :param query: The content of the user's chat message. Can be None to generate a completion without a user
            prompt.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        async for elem in self._full_round(query, _kani_is_stream=False, **kwargs):
            yield elem

    async def full_round_str(
        self,
        query: QueryType,
        message_formatter: Callable[[ChatMessage], str | None] = assistant_message_contents,
        **kwargs,
    ) -> AsyncIterable[str]:
        """Like :meth:`full_round`, but each yielded element is a str rather than a ChatMessage.

        :param query: The content of the user's chat message.
        :param message_formatter: A function that returns a string to yield for each message. By default,
            ``full_round_str`` yields the content of each assistant message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        async for message in self.full_round(query, **kwargs):
            if text := message_formatter(message):
                yield text

    async def full_round_stream(self, query: QueryType, **kwargs) -> AsyncIterable[StreamManager]:
        """
        Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields a stream of tokens for each non-user ChatMessage created during the round.

        To consume tokens from a stream, use this class as so::

            async for stream in ai.full_round_stream("What is the airspeed velocity of an unladen swallow?"):
                async for token in stream:
                    print(token, end="")
                msg = await stream.message()

        Each :class:`.StreamManager` object yielded by this method contains a :attr:`.StreamManager.role` attribute
        that can be used to determine if a message is from the engine or a function call. This attribute will be
        available *before* iterating over the stream.

        The arguments are the same as :meth:`full_round`.
        """
        async for elem in self._full_round(query, _kani_is_stream=True, **kwargs):
            yield elem

    # ==== helpers ====
    @property
    def always_len(self) -> int:
        """Returns the number of tokens that will always be reserved.

        (e.g. for system prompts, always included messages, the engine, and the response).
        """
        return (
            sum(self.message_token_len(m) for m in self.always_included_messages)
            + self.engine.token_reserve
            + self.engine.function_token_reserve(list(self.functions.values()))
            + self.desired_response_tokens
        )

    def message_token_len(self, message: ChatMessage):
        """Returns the number of tokens used by a given message."""
        return self.engine.message_len(message)

    async def get_model_completion(self, include_functions: bool = True, **kwargs) -> BaseCompletion:
        """Get the model's completion with the current chat state.

        Compared to :meth:`chat_round` and :meth:`full_round`, this lower-level method does not save the model's reply
        to the chat history or mutate the chat state; it is intended to help with logging or to repeat a call multiple
        times.

        :param include_functions: Whether to pass this kani's function definitions to the engine.
        :param kwargs: Arguments to pass to the model engine.
        """
        # get the current chat state
        messages = await self.get_prompt()
        # log it (message_log includes the number of messages sent and the last message)
        n_messages = len(messages)
        if n_messages == 0:
            message_log.debug("[0]>>> [requested completion with no prompt]")
        else:
            message_log.debug(f"[{n_messages}]>>> {messages[-1]}")

        # get the model's completion at the given state
        if include_functions:
            completion = await self.engine.predict(messages=messages, functions=list(self.functions.values()), **kwargs)
        else:
            completion = await self.engine.predict(messages=messages, **kwargs)

        message_log.debug(f"<<< {completion.message}")
        return completion

    async def get_model_stream(self, include_functions: bool = True, **kwargs) -> AsyncIterable[str | BaseCompletion]:
        """Get the model's completion with the current chat state as a stream.
        This is a low-level method like :meth:`get_model_completion` but for streams.
        """
        messages = await self.get_prompt()
        n_messages = len(messages)
        if n_messages == 0:
            message_log.debug("[0]>>> [requested completion with no prompt]")
        else:
            message_log.debug(f"[{n_messages}]>>> {messages[-1]}")

        # get the model's completion at the given state
        if include_functions:
            stream = self.engine.stream(messages=messages, functions=list(self.functions.values()), **kwargs)
        else:
            stream = self.engine.stream(messages=messages, **kwargs)

        message_log.debug(f"<<< STREAM...")
        async for elem in stream:
            yield elem

    # ==== overridable methods ====
    async def get_prompt(self) -> list[ChatMessage]:
        """
        Called each time before asking the LM engine for a completion to generate the chat prompt.
        Returns a list of messages such that the total token count in the messages is less than
        ``(self.max_context_size - self.desired_response_tokens)``.

        Always includes the system prompt plus any always_included_messages at the start of the prompt.

        You may override this to get more fine-grained control over what is exposed in the model's memory at any given
        call.
        """
        always_len = self.always_len
        remaining = max_size = self.max_context_size - always_len
        total_tokens = 0
        to_keep = 0  # messages to keep from the end of chat history
        for message in reversed(self.chat_history):
            # get and check the message's length
            message_len = self.message_token_len(message)
            if message_len > max_size:
                func_help = (
                    ""
                    if message.role != ChatRole.FUNCTION
                    else "You may set `auto_truncate` in the @ai_function to automatically truncate long responses.\n"
                )
                raise MessageTooLong(
                    "The chat message's size is longer than the allowed context window (after including system"
                    " messages, always included messages, and desired response tokens).\n"
                    f"{func_help}Content: {message.text[:100]}..."
                )
            # see if we can include it
            remaining -= message_len
            if remaining >= 0:
                total_tokens += message_len
                to_keep += 1
            else:
                break
        log.debug(
            f"get_prompt() returned {always_len + total_tokens} tokens ({always_len} always) in"
            f" {len(self.always_included_messages) + to_keep} messages"
            f" ({len(self.always_included_messages)} always)"
        )
        if not to_keep:
            return self.always_included_messages
        return self.always_included_messages + self.chat_history[-to_keep:]

    async def do_function_call(self, call: FunctionCall, tool_call_id: str = None) -> FunctionCallResult:
        """Resolve a single function call.

        By default, any exception raised from this method will be an instance of a :class:`.FunctionCallException`.

        You may implement an override to add instrumentation around function calls (e.g. tracking success counts
        for varying prompts). See :doc:`/customization/function_call`.

        :param call: The name of the function to call and arguments to call it with.
        :param tool_call_id: The ``tool_call_id`` to set in the returned FUNCTION message.
        :returns: A :class:`.FunctionCallResult` including whose turn it is next and the message with the result of the
            function call.
        :raises NoSuchFunction: The requested function does not exist.
        :raises WrappedCallException: The function raised an exception.
        """
        log.debug(f"Model requested call to {call.name} with data: {call.arguments!r}")
        # get func
        f = self.functions.get(call.name)
        if not f:
            raise NoSuchFunction(call.name)
        # call it
        try:
            result = await f(**call.kwargs)
            result_str = str(result)
            log.debug(f"{f.name} responded with data: {result_str!r}")
        except Exception as e:
            raise WrappedCallException(f.auto_retry, e) from e
        msg = ChatMessage.function(f.name, result_str, tool_call_id=tool_call_id)
        # if we are auto truncating, check and see if we need to
        if f.auto_truncate is not None:
            message_len = self.message_token_len(msg)
            if message_len > f.auto_truncate:
                log.warning(
                    f"The content returned by {f.name} is too long ({message_len} > {f.auto_truncate} tokens), auto"
                    " truncating..."
                )
                msg = self._auto_truncate_message(msg, max_len=f.auto_truncate)
                log.debug(f"Auto truncate returned {self.message_token_len(msg)} tokens.")
        return FunctionCallResult(is_model_turn=f.after == ChatRole.ASSISTANT, message=msg)

    async def handle_function_call_exception(
        self, call: FunctionCall, err: FunctionCallException, attempt: int, tool_call_id: str = None
    ) -> ExceptionHandleResult:
        """Called when a function call raises an exception.

        By default, returns a message telling the LM about the error and allows a retry if the error
        is recoverable and there are remaining retry attempts.

        You may implement an override to customize the error prompt, log the error, or use custom retry logic.
        See :doc:`/customization/function_exception`.

        :param call: The :class:`.FunctionCall` the model was attempting to make.
        :param err: The error the call raised. Usually this is :class:`.NoSuchFunction` or
            :class:`.WrappedCallException`, although it may be any exception raised by :meth:`do_function_call`.
        :param attempt: The attempt number for the current call (0-indexed).
        :param tool_call_id: The ``tool_call_id`` to set in the returned FUNCTION message.
        :returns: A :class:`.ExceptionHandleResult` detailing whether the model should retry and the message to add to
            the chat history.
        """
        # log the exception here
        log.debug(f"Call to {call.name} raised an exception: {err}")
        # tell the model what went wrong
        if isinstance(err, NoSuchFunction):
            msg = ChatMessage.function(
                name=None,
                content=f"The function {err.name!r} is not defined. Only use the provided functions.",
                tool_call_id=tool_call_id,
            )
        else:
            # but if it's a user function error, we want to raise it
            log.error(f"Call to {call.name} raised an exception: {err}", exc_info=err)
            msg = ChatMessage.function(call.name, str(err), tool_call_id=tool_call_id)

        return ExceptionHandleResult(should_retry=attempt < self.retry_attempts and err.retry, message=msg)

    async def add_to_history(self, message: ChatMessage):
        """Add the given message to the chat history.

        You might want to override this to log messages to an external or control how messages are saved to the chat
        session's memory. By default, this appends to :attr:`.chat_history`.
        """
        # this is async even though the default impl is sync because users might conceivably want to perform I/O
        self.chat_history.append(message)

    # ==== utility methods ====
    def save(self, fp: PathLike, **kwargs):
        """Save the chat state of this kani to a JSON file. This will overwrite the file if it exists!

        :param fp: The path to the file to save.
        :param kwargs: Additional arguments to pass to Pydantic's ``model_dump_json``.
        """
        data = SavedKani(always_included_messages=self.always_included_messages, chat_history=self.chat_history)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(**kwargs))

    def load(self, fp: PathLike, **kwargs):
        """Load chat state from a JSON file into this kani. This will overwrite any existing chat state!

        :param fp: The path to the file containing the chat state.
        :param kwargs: Additional arguments to pass to Pydantic's ``model_validate_json``.
        """
        with open(fp, encoding="utf-8") as f:
            data = f.read()
        state = SavedKani.model_validate_json(data, **kwargs)
        self.always_included_messages = state.always_included_messages
        self.chat_history = state.chat_history

    # ==== internals ====
    def _auto_truncate_message(self, msg: ChatMessage, max_len: int) -> ChatMessage:
        """Mutate the provided message until it is less than *max_len* tokens long."""
        full_text = msg.text
        if not full_text:
            return msg  # idk how this could happen
        for chunk_divider in ("\n\n", "\n", ". ", ", ", " "):
            # chunk the text
            text = ""
            last_msg = None
            chunks = full_text.split(chunk_divider)
            for idx, chunk in enumerate(chunks):
                # fit in as many chunks as possible
                if idx:
                    text += chunk_divider
                text += chunk
                # when it's too long...
                msg = msg.copy_with(text=text + "...")
                if self.message_token_len(msg) > max_len:
                    # if we have some text, return it
                    if last_msg:
                        return last_msg
                    # otherwise, we need to split into smaller chunks
                    break
                # otherwise, continue
                last_msg = msg
        # if we get here and have no content, chop it to the first max_len characters
        log.warning(
            "Auto truncate could not find an appropriate place to chunk the text. The returned value will be the first"
            f" {max_len} characters."
        )
        return msg.copy_with(text=full_text[: max_len - 3] + "...")
