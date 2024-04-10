import json

from kani import AIFunction
from kani.models import ChatMessage, ChatRole
from kani.prompts.pipeline import PromptPipeline
from kani.prompts.steps import Apply, ConversationFmt, MergeConsecutive, Remove

# ==== default prompts ====
# fmt: off
DEFAULT_SYSTEM_PROMPT = "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."

DEFAULT_PREAMBLE = """# Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
"""
DEFAULT_TASK = """## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."""

DEFAULT_TOOL_PROMPT = '''

## Available Tools
Here is a list of tools that you have available to you:

{user_functions}

```python
def directly_answer() -> List[Dict]:
    """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
    """
    pass
```
'''
DEFAULT_TOOL_INSTRUCTIONS = """Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
```json
[
    {
        "tool_name": title of the tool in the specification,
        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
    }
]
```"""

DEFAULT_RAG_INSTRUCTIONS_ACC = """Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0."""
DEFAULT_RAG_INSTRUCTIONS_FAST = """Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0."""
# fmt: on

# ==== no tool calling ====
COMMAND_R_PIPELINE = (
    PromptPipeline()
    .conversation_fmt(
        prefix="<BOS_TOKEN>",
        generation_suffix="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        user_prefix="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        user_suffix="<|END_OF_TURN_TOKEN|>",
        assistant_prefix="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        assistant_suffix="<|END_OF_TURN_TOKEN|>",
        assistant_suffix_if_last="",
        system_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        system_suffix="<|END_OF_TURN_TOKEN|>",
        function_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\n<results>\n",
        function_suffix="\n</results><|END_OF_TURN_TOKEN|>",
    )
)  # fmt: skip
"""The pipeline to use when interfacing with Command R without tools defined."""


# ==== tool calling (function not last) ====
def function_result_joiner(msgs):
    contents = []
    for idx, msg in enumerate(msgs):
        contents.append(f"Document: {idx}\n{msg.text}")
    return "\n\n".join(contents)


def tool_call_formatter(msg: ChatMessage):
    if msg.tool_calls:
        text = msg.text + "\n" if msg.text else ""
        tool_calls = json.dumps(
            [{"tool_name": tc.function.name, "parameters": tc.function.kwargs} for tc in msg.tool_calls],
            indent=4,
        )
        return f"{text}Action: ```json\n{tool_calls}\n```"
    else:
        return (  # is the EOT/SOT token doing weird stuff here?
            'Action: ```json\n[\n    {\n        "tool_name": "directly_answer",\n        "parameters": {}\n'
            f"    }}\n]\n```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{msg.text}"
        )


def build_tool_pipeline(
    *, include_function_calls=True, include_function_results=True, tool_instructions=DEFAULT_TOOL_INSTRUCTIONS
):
    """
    The pipeline to use when interfacing with Command R WITH tools defined. Use this pipeline if the last message is
    NOT a FUNCTION message.

    :param include_function_calls: Whether to include previous turns' function calls or just the model's answers.
    :param include_function_results: Whether to include the results of previous turns' function calls in the context.
    :param tool_instructions: The system prompt to send just before the model's generation turn that includes
        instructions on the format to generate tool calls in. Generally you shouldn't change this.
    """

    steps = []

    # format function calls with an Action: prefix; otherwise do a directly_answer call
    if include_function_calls:
        steps.append(Apply(tool_call_formatter, role=ChatRole.ASSISTANT))

    # keep function results around as SYSTEM messages
    if include_function_results:
        steps.append(MergeConsecutive(role=ChatRole.FUNCTION, joiner=function_result_joiner))
    else:
        steps.append(Remove(role=ChatRole.FUNCTION))

    steps.append(
        ConversationFmt(
            prefix="<BOS_TOKEN>",
            generation_suffix=(
                f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{tool_instructions}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            ),
            user_prefix="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            user_suffix="<|END_OF_TURN_TOKEN|>",
            assistant_prefix="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            assistant_suffix="<|END_OF_TURN_TOKEN|>",
            assistant_suffix_if_last="",
            system_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            system_suffix="<|END_OF_TURN_TOKEN|>",
            function_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\n<results>\n",
            function_suffix="\n</results><|END_OF_TURN_TOKEN|>",
        )
    )

    return PromptPipeline(steps)


# ==== tool calling (function last) ====
def build_rag_pipeline(*, include_previous_results=True, rag_instructions=DEFAULT_RAG_INSTRUCTIONS_ACC):
    """
    The pipeline to use when interfacing with Command R WITH tools defined. Use this pipeline if the last message IS a
    FUNCTION message.

    :param include_previous_results: Include previous turns' results in the chat history.
    :param rag_instructions: The system prompt to send just before the model's generation turn that includes
        instructions on the format to generate the result in. Can be None to only generate a model turn. Defaults
        to the "accurate" grounded RAG prompt (``from kani.prompts.impl.cohere import DEFAULT_RAG_INSTRUCTIONS_ACC``).
    """

    def remover(m, is_last):
        return None if is_last and not include_previous_results else m

    return (
        PromptPipeline()
        .merge_consecutive(role=ChatRole.FUNCTION, joiner=function_result_joiner)
        # remove all but the last function message
        .apply(remover, role=ChatRole.FUNCTION)
        .conversation_fmt(
            prefix="<BOS_TOKEN>",
            generation_suffix=(
                f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{rag_instructions}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
                if rag_instructions
                else "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            ),
            user_prefix="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            user_suffix="<|END_OF_TURN_TOKEN|>",
            assistant_prefix="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            assistant_suffix="<|END_OF_TURN_TOKEN|>",
            assistant_suffix_if_last="",
            system_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            system_suffix="<|END_OF_TURN_TOKEN|>",
            function_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\n<results>\n",
            function_suffix="\n</results><|END_OF_TURN_TOKEN|>",
        )
    )


# ==== helpers ====
def function_prompt(f: AIFunction) -> str:
    params = f.get_params()

    # build params
    param_parts = []
    for param in params:
        default = ""
        if param.default:
            default = f" = {param.default}"
        param_parts.append(f"{param.name}: {param.type}{default}")
    params_str = ", ".join(param_parts)

    # build docstring
    args = ""
    if len(params):
        args = f"\n\n    Args:\n        "
        doc_params = []
        for param in params:
            desc = f": {param.description}" if param.description else ""
            doc_params.append(f"{param.name} ({param.type}){desc}")
        args += "\n        ".join(doc_params)

    # return
    return f'```python\ndef {f.name}({params_str}) -> List[Dict]:\n    """{f.desc}{args}\n    """\n    pass\n```'
