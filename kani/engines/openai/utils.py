class DottableDict(dict):
    # compatibility for changing the returned extra type from a Pydantic model to a dict
    def __getattr__(self, item):
        return self[item]
