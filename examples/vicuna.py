from kani import Kani, chat_in_terminal
from kani.engines.huggingface.vicuna import VicunaEngine

engine = VicunaEngine()
ai = Kani(
    engine,
    system_prompt=(
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed,"
        " and polite answers to the user's questions."
    ),
)
if __name__ == '__main__':
    chat_in_terminal(ai)
