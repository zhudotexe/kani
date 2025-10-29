# End-to-End Tests

E2E tests cache the response from the LLM as close to the low level as possible (e.g., mocking API responses/calls to HF
.generate). We want to make sure the requests Kani is sending to the lower level inference stack don't change.

E2E tests need to be hydrated using real API calls/GPUs at least once.

```shell
KANI_E2E_HYDRATE=api pytest -m e2e -s
KANI_E2E_HYDRATE=local pytest -m e2e -s
KANI_E2E_HYDRATE=llamacpp pytest -m e2e -s
```

## Writing Tests

Use parameterization to request models & request model capabilities

## Testing additional models

for HF, change the models to test constant in conftest.py
