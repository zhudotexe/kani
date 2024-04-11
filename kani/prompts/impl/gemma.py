from kani.models import ChatRole
from kani.prompts.pipeline import PromptPipeline

GEMMA_PIPELINE = (
    PromptPipeline()
    .translate_role(
        role=(ChatRole.SYSTEM, ChatRole.FUNCTION),
        to=ChatRole.USER,
        warn=(
            "The Gemma prompt format does not natively support the SYSTEM or FUNCTION roles. These messages will be"
            " sent to the model as USER messages."
        ),
    )
    .conversation_fmt(
        prefix="<bos>",
        generation_suffix="<start_of_turn>model\n",
        user_prefix="<start_of_turn>user\n",
        user_suffix="<end_of_turn>\n",
        assistant_prefix="<start_of_turn>model\n",
        assistant_suffix="<end_of_turn>\n",
        assistant_suffix_if_last="",
    )
)

# from https://huggingface.co/google/gemma-1.1-2b-it/blob/main/tokenizer_config.json:
# {{ bos_token }}
# {% if messages[0]['role'] == 'system' %}
#   {{ raise_exception('System role not supported') }}
# {% endif %}
# {% for message in messages %}
#   {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
#       {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
#   {% endif %}
#   {% if (message['role'] == 'assistant') %}
#       {% set role = 'model' %}
#   {% else %}
#       {% set role = message['role'] %}
#   {% endif %}
#   {{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}
# {% endfor %}
# {% if add_generation_prompt %}
#   {{'<start_of_turn>model\n'}}
# {% endif %}
