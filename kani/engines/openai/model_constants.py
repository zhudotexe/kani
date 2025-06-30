# https://platform.openai.com/docs/models
CONTEXT_SIZES_BY_PREFIX = [
    ("gpt-3.5-turbo-instruct", 4096),
    ("gpt-3.5-turbo-0613", 4096),
    ("gpt-3.5-turbo", 16385),
    # o1, o3, o4
    ("o1", 200000),
    ("o3", 200000),
    ("o4", 200000),
    # gpt-4.1
    ("gpt-4.1", 1047576),
    # gpt-4o
    ("gpt-4o", 128000),
    ("chatgpt-4o", 128000),
    # gpt-4-turbo models aren't prefixed differently...
    ("gpt-4-1106", 128000),
    ("gpt-4-0125", 128000),
    ("gpt-4-vision", 128000),
    ("gpt-4-turbo", 128000),
    ("gpt-4-32k", 32768),
    ("gpt-4", 8192),
    # fine-tunes
    ("ft:gpt-3.5-turbo-instruct", 4096),
    ("ft:gpt-3.5-turbo-0613", 4096),
    ("ft:gpt-3.5-turbo", 16385),
    ("ft:gpt-4-32k", 32768),
    ("ft:gpt-4", 8192),
    # completion models
    ("babbage-002", 16384),
    ("davinci-002", 16384),
    # catch-all
    ("", 2048),  # e.g. aba/babbage/curie/davinci
]

# ===== multimodal =====
# ---- images ----
# model_id -> multiplier
MM_IMAGE_LOW_COST_SCALE = {
    "gpt-4.1-mini": 1.62,
    "gpt-4.1-nano": 2.46,
    "o4-mini": 1.72,
}
# model_id -> (base, per_patch)
MM_IMAGE_OLD_SCALE = {
    "gpt-4o": (85, 170),
    "gpt-4.1": (85, 170),
    "gpt-4.5": (85, 170),
    "gpt-4o-mini": (2833, 5667),
    "o1": (75, 150),
    "o3": (75, 150),
    None: (85, 170),  # default
}
