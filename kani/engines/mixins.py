import json

from kani.ai_function import AIFunction
from kani.models import ChatMessage


class TokenCached:
    """
    Mixin to cache token counts for individual messages or prompts.

    Use ``self.get_cached_prompt_len`` and ``self.set_cached_prompt_len`` to manage the cache.
    The caches are unbounded since they are very lightweight (an int -> int dict).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_token_cache = {}
        self._prompt_token_cache = {}

    # ==== prompt-level ====
    def prompt_cache_key(self, messages: list[ChatMessage], functions: list[AIFunction] | None, **kwargs):
        # for now, we won't handle kwargs
        if kwargs:
            return None
        # ((...message keys), (...(json schemas)))
        if not functions:
            function_schemas = None
        else:
            function_schemas = tuple(json.dumps(f.json_schema) for f in functions)
        return hash((tuple(self.message_cache_key(m) for m in messages), function_schemas))

    def get_cached_prompt_len(self, messages: list[ChatMessage], functions: list[AIFunction], **kwargs) -> int | None:
        cache_key = self.prompt_cache_key(messages, functions, **kwargs)
        if not cache_key:
            return None
        if cache_key in self._prompt_token_cache:
            return self._prompt_token_cache[cache_key]
        return None

    def set_cached_prompt_len(self, messages: list[ChatMessage], functions: list[AIFunction], length: int, **kwargs):
        cache_key = self.prompt_cache_key(messages, functions, **kwargs)
        if not cache_key:
            return
        self._prompt_token_cache[cache_key] = length

    # ==== message-level ====
    def message_cache_key(self, message: ChatMessage):
        # (role, content, tool calls)

        # we'll use msgpart identity for the hash here since we'll always have a ref as long as it's in a message
        # history
        hashable_content = tuple(part if isinstance(part, str) else id(part) for part in message.parts)

        # use (name, args) for tool calls
        if message.tool_calls is not None:
            hashable_tool_calls = tuple((tc.function.name, tc.function.arguments) for tc in message.tool_calls)
        else:
            hashable_tool_calls = message.tool_calls

        return hash((message.role, hashable_content, hashable_tool_calls))

    def get_cached_message_len(self, message: ChatMessage) -> int | None:
        # use cache
        cache_key = self.message_cache_key(message)
        if cache_key in self._message_token_cache:
            return self._message_token_cache[cache_key]
        return None

    def set_cached_message_len(self, message: ChatMessage, length: int):
        cache_key = self.message_cache_key(message)
        self._message_token_cache[cache_key] = length
