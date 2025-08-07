from transformers import AutoModelForCausalLM, AutoTokenizer

# define a simple tool calling prompt
MESSAGES_HF_SHORT = [
    {
        "role": "system",
        "content": "IMPORTANT: Always tell the your plan before starting any tool calls.",
    },
    {"role": "user", "content": "What's the weather in Tokyo?"},
]


def get_weather(location: str):
    """
    Get the weather at a location.

    Args:
        location: The location
    """


# print the prompt to verify
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
print(tok.apply_chat_template(MESSAGES_HF_SHORT, tokenize=False, tools=[get_weather], add_generation_prompt=True))

# generate a completion from the model
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-120b", device_map="auto", torch_dtype="auto")
output = model.generate(
    tok.apply_chat_template(MESSAGES_HF_SHORT, tools=[get_weather], add_generation_prompt=True, return_tensors="pt").to(
        "cuda"
    ),
    max_new_tokens=2048,
    temperature=1,
    top_p=1,
    eos_token_id=[200002, 199999, 200012],
)

# observe the preamble in the commentary channel
content = tok.decode(output[0])
print(content)
