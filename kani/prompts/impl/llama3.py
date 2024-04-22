"""Common builder for the LLaMAv3-chat prompt."""

from kani.models import ChatRole
from kani.prompts.pipeline import PromptPipeline

LLAMA3_PIPELINE = (
    PromptPipeline()
    .translate_role(
        role=ChatRole.FUNCTION,
        to=ChatRole.USER,
        warn=(
            "The Llama 3 prompt format does not natively support the FUNCTION role. These messages will be"
            " sent to the model as USER messages."
        ),
    )
    .conversation_fmt(
        prefix="<|begin_of_text|>",
        generation_suffix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|>",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_suffix="<|eot_id|>",
        assistant_suffix_if_last="",
        system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
        system_suffix="<|eot_id|>",
    )
)  # fmt: skip

# from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# {% set loop_messages = messages %}
# {% for message in loop_messages %}
#   {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}
#   {% if loop.index0 == 0 %}
#       {% set content = bos_token + content %}
#   {% endif %}
#   {{ content }}
# {% endfor %}
# {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
