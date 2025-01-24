from kani import Kani, chat_in_terminal
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl import LLAMA2_PIPELINE

model_id = "apple/OpenELM-270M-Instruct"


engine = HuggingEngine(
    model_id=model_id,
    prompt_pipeline=LLAMA2_PIPELINE,
    tokenizer_kwargs=dict(trust_remote_code=True),
    model_load_kwargs=dict(trust_remote_code=True),
)

ai = Kani(engine)

if __name__ == "__main__":
    chat_in_terminal(ai)
