CONTEXT_SIZES_BY_PREFIX = [
    ("claude-sonnet-4-5-20250929", 200000),
    ("claude-haiku-4-5-20251001", 200000),
    ("claude-opus-4-5-20251101", 200000),
    ("claude-sonnet-4-6", 1000000),
    ("claude-haiku-5-5", 1000000),
    ("claude-sonnet-5", 1000000),
    ("claude-opus-4-8", 1000000),
    ("claude-opus-4-7", 1000000),
    ("claude-opus-4-6", 1000000),
    ("claude-fable-5", 1000000),
    ("claude-opus-5", 1000000),
    ("", 200000),
]


if __name__ == "__main__":
    # helper script to generate this list from the API
    import anthropic

    client = anthropic.Client()

    # get list of all models
    models = []
    after_id = anthropic.omit
    while True:
        page = client.models.list(after_id=after_id)
        models.extend(page.data)
        if not page.has_more:
            break
        after_id = page.last_id

    # remove models that are covered by a more generic version later
    sorted_len = sorted(models, key=lambda m: len(m.id), reverse=True)
    for model in sorted_len.copy():
        for other_model in sorted_len:
            if model.id == other_model.id:
                continue
            if model.id.startswith(other_model.id):
                if model.max_input_tokens == other_model.max_input_tokens:
                    sorted_len.remove(model)
                break

    # add the default
    print([(m.id, m.max_input_tokens) for m in sorted_len] + [("", 200000)])
