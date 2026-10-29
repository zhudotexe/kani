CONTEXT_SIZES_BY_PREFIX = [
    ("gemini-2.5-computer-use-preview-10-2025", 131072),
    ("deep-research-max-preview-04-2026", 131072),
    ("deep-research-pro-preview-12-2025", 131072),
    ("deep-research-preview-04-2026", 131072),
    ("gemini-2.5-flash-preview-tts", 8192),
    ("gemini-3.1-flash-tts-preview", 8192),
    ("gemini-robotics-er-2-preview", 131072),
    ("gemini-3.1-flash-lite-image", 65536),
    ("antigravity-preview-05-2026", 1048576),
    ("antigravity-preview-09-2026", 1048576),
    ("gemini-2.5-pro-preview-tts", 8192),
    ("antigravity-preview-latest", 1048576),
    ("gemini-omni-flash-preview", 131072),
    ("gemini-3.8-flash-lite-tts", 8192),
    ("gemini-flash-lite-latest", 1048576),
    ("nano-banana-pro-preview", 131072),
    ("gemini-2.5-flash-image", 32768),
    ("gemini-3-flash-preview", 1048576),
    ("gemini-3.1-pro-preview", 1048576),
    ("gemini-3.1-flash-image", 65536),
    ("gemini-nano-banana-2.1", 65536),
    ("gemini-3.1-flash-lite", 1048576),
    ("gemini-omni-1.1-flash", 131072),
    ("gemini-3.5-transcribe", 98304),
    ("lyria-3-clip-preview", 1048576),
    ("gemini-3.8-flash-tts", 8192),
    ("gemini-flash-latest", 1048576),
    ("lyria-3-pro-preview", 1048576),
    ("gemma-4-26b-a4b-it", 262144),
    ("gemini-3-pro-image", 131072),
    ("gemini-pro-latest", 1048576),
    ("gemini-2.5-flash", 1048576),
    ("gemini-3.5-flash", 1048576),
    ("gemini-3.6-flash", 1048576),
    ("gemini-3.7-flash", 1048576),
    ("gemini-3.8-flash", 1048576),
    ("gemini-2.5-pro", 1048576),
    ("gemma-4-31b-it", 262144),
    ("gemini-4-argon", 1048576),
    ("lyria-3.5", 1048576),
    ("", 1048576),
]


if __name__ == "__main__":
    # helper script to generate this list from the API
    from google import genai

    client = genai.Client()

    # get list of all models
    models = []
    for m in client.models.list():
        for action in m.supported_actions:
            if action == "generateContent":
                models.append((m.name.removeprefix("models/"), m.input_token_limit))

    # remove models that are covered by a more generic version later
    sorted_len = sorted(models, key=lambda m: len(m[0]), reverse=True)
    for model, ctxlen in sorted_len.copy():
        for other_model, other_ctxlen in sorted_len:
            if model == other_model:
                continue
            if model.startswith(other_model):
                if ctxlen == other_ctxlen:
                    sorted_len.remove((model, ctxlen))
                break

    # add the default
    sorted_len.append(("", 1048576))

    print(sorted_len)
