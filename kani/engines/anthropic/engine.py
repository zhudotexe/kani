import os
import warnings

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies, PromptError
from kani.models import ChatMessage, ChatRole
from ..base import BaseEngine, Completion

try:
    from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic

    # anthropic async client loads a small json file using anyio for some reason; hook into the underlying loader
    # noinspection PyProtectedMember
    from anthropic._tokenizers import sync_get_tokenizer
except ImportError as e:
    raise MissingModelDependencies(
        'The AnthropicEngine requires extra dependencies. Please install kani with "pip install kani[anthropic]".'
    ) from None

CONTEXT_SIZES_BY_PREFIX = [
    ("claude-2.1", 200000),
    ("", 100000),
]


class AnthropicEngine(BaseEngine):
    """Engine for using the Anthropic API.

    This engine supports all Claude models. See https://docs.anthropic.com/claude/docs/getting-access-to-claude for
    information on accessing the Claude API.

    See https://docs.anthropic.com/claude/reference/selecting-a-model for a list of available models.
    """

    token_reserve = 4  # each prompt ends with \n\nAssistant:

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-2.1",
        max_tokens_to_sample: int = 512,
        max_context_size: int = None,
        *,
        retry: int = 2,
        api_base: str = None,
        headers: dict = None,
        client: AsyncAnthropic = None,
        **hyperparams,
    ):
        """
        :param api_key: Your Anthropic API key. By default, the API key will be read from the `ANTHROPIC_API_KEY`
            environment variable.
        :param model: The id of the model to use (e.g. "claude-2.1", "claude-instant-1.2").
        :param max_tokens_to_sample: The maximum number of tokens to sample at each generation (defaults to 450).
            Generally, you should set this to the same number as your Kani's ``desired_response_tokens``.
        :param max_context_size: The maximum amount of tokens allowed in the chat prompt. If None, uses the given
            model's full context size.
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 2).
        :param api_base: The base URL of the Anthropic API to use.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of ``anthropic.AsyncAnthropic`` (for reusing the same client in multiple engines).
            You must specify exactly one of (api_key, client). If this is passed the ``retry``, ``api_base``,
            and ``headers`` params will be ignored.
        :param hyperparams: Any additional parameters to pass to the underlying API call (see
            https://docs.anthropic.com/claude/reference/complete_post).
        """
        if api_key and client:
            raise ValueError("You must supply no more than one of (api_key, client).")
        if api_key is None and client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "You must supply an `api_key`, `client`, or set the `OPENAI_API_KEY` environment variable to use"
                    " the OpenAIEngine."
                )
        if max_context_size is None:
            max_context_size = next(size for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix))
        self.client = client or AsyncAnthropic(
            api_key=api_key, max_retries=retry, base_url=api_base, default_headers=headers
        )
        self.model = model
        self.max_tokens_to_sample = max_tokens_to_sample
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams
        self.tokenizer = sync_get_tokenizer()

    def message_len(self, message: ChatMessage) -> int:
        # human messages are prefixed with `\n\nHuman: ` and assistant with `\n\nAssistant:`
        if message.role == ChatRole.USER:
            mlen = 5
        elif message.role == ChatRole.ASSISTANT:
            mlen = 4
        else:
            mlen = 2  # we'll prepend system/function messages with \n\n as a best-effort case

        if message.text:
            mlen += len(self.tokenizer.encode(message.text).ids)
        return mlen

    @staticmethod
    def build_prompt(messages: list[ChatMessage]):
        # Claude prompts must start with a human message
        first_human_idx = next((i for i, m in enumerate(messages) if m.role == ChatRole.USER), None)
        if first_human_idx is None:
            raise PromptError("Prompts to Anthropic models must contain at least one USER message.")

        # and make sure the system messages are included
        last_system_idx = next((i for i, m in enumerate(messages) if m.role != ChatRole.SYSTEM), None)
        if last_system_idx:
            out = ["\n\n".join(m.text for m in messages[:last_system_idx])]
        else:
            out = []

        for idx, message in enumerate(messages[first_human_idx:]):
            if message.role == ChatRole.USER:
                out.append(f"{HUMAN_PROMPT} {message.text}")
            elif message.role == ChatRole.ASSISTANT:
                out.append(f"{AI_PROMPT} {message.text}")
            else:
                warnings.warn(
                    f"Encountered a {message.role} message in the middle of the prompt - Anthropic models expect an"
                    " optional SYSTEM message followed by alternating USER and ASSISTANT messages. Appending the"
                    " content to the prompt..."
                )
                out.append(f"\n\n{message.text}")
        return "".join(out) + AI_PROMPT

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        prompt = self.build_prompt(messages)
        completion = await self.client.completions.create(
            model=self.model,
            max_tokens_to_sample=self.max_tokens_to_sample,
            prompt=prompt,
            **self.hyperparams,
            **hyperparams,
        )
        return Completion(message=ChatMessage.assistant(completion.completion.strip()))

    async def close(self):
        await self.client.close()
