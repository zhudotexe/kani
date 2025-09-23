CONTEXT_SIZES_BY_PREFIX = [
    ("gemini-2.0-flash-preview-image-generation", 32768),
    ("learnlm-2.0-flash-experimental", 1048576),
    ("gemini-2.5-flash-image-preview", 32768),
    ("gemini-2.5-flash-preview-tts", 8192),
    ("gemini-2.5-pro-preview-tts", 8192),
    ("gemini-2.0-pro-exp", 1048576),
    ("gemini-1.5-flash", 1000000),
    ("gemini-2.5-flash", 1048576),
    ("gemini-2.0-flash", 1048576),
    ("gemini-exp-1206", 1048576),
    ("gemma-3n-e4b-it", 8192),
    ("gemma-3n-e2b-it", 8192),
    ("gemini-1.5-pro", 2000000),
    ("gemini-2.5-pro", 1048576),
    ("gemma-3-12b-it", 32768),
    ("gemma-3-27b-it", 131072),
    ("gemma-3-1b-it", 32768),
    ("gemma-3-4b-it", 32768),
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
