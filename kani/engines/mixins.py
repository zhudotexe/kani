from kani.models import ChatMessage


class TokenCached:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_cache = {}

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
        if cache_key in self.token_cache:
            return self.token_cache[cache_key]

    def set_cached_message_len(self, message: ChatMessage, length: int):
        cache_key = self.message_cache_key(message)
        self.token_cache[cache_key] = length
