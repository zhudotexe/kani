import inspect
import json
import logging
import re
from collections import namedtuple

from kani import AIFunction
from kani.engines import Completion
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
from kani.prompts.pipeline import PromptPipeline
from kani.prompts.steps import Apply, ConversationFmt, MergeConsecutive, Remove

log = logging.getLogger(__name__)

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


def function_result_joiner(msgs):
    contents = []
    for idx, msg in enumerate(msgs):
        contents.append(f"Document: {idx}\n{msg.text}")
    return "\n\n".join(contents)


def tool_call_formatter(msg: ChatMessage) -> str:
    if msg.tool_calls:
        text = msg.text + "\n" if msg.text else ""
        tool_calls = json.dumps(
            [{"tool_name": tc.function.name, "parameters": tc.function.kwargs} for tc in msg.tool_calls],
            indent=4,
        )
        return f"{text}Action: ```json\n{tool_calls}\n```"
    # else:
    #     return (  # is the EOT/SOT token doing weird stuff here?
    #         'Action: ```json\n[\n    {\n        "tool_name": "directly_answer",\n        "parameters": {}\n'
    #         f"    }}\n]\n```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{msg.text}"
    #     )
    return msg.content


def build_pipeline(
    *,
    include_function_calls=True,
    include_all_function_results=True,
    include_last_function_result=True,
    instruction_suffix=None,
):
    """
    :param include_function_calls: Whether to include previous turns' function calls or just the model's answers.
    :param include_all_function_results: Whether to include the results of all previous turns' function calls in the
        context.
    :param include_last_function_result: If *include_all_function_results* is False, whether to include just the last
        function call's result (useful for RAG).
    :param instruction_suffix: The system prompt to send just before the model's generation turn that includes
        instructions on the format to generate the result in. Can be None to only generate a model turn.
        For tool calling, this should be the DEFAULT_TOOL_INSTRUCTIONS.
        For RAG, this should be DEFAULT_RAG_INSTRUCTIONS_ACC or DEFAULT_RAG_INSTRUCTIONS_FAST.
    """

    steps = []

    # format function calls with an Action: prefix
    if include_function_calls:

        def apply_tc_format(msg):
            msg.content = tool_call_formatter(msg)
            return msg

        steps.append(Apply(apply_tc_format, role=ChatRole.ASSISTANT))
    else:
        steps.append(Remove(role=ChatRole.ASSISTANT, predicate=lambda msg: msg.content is None))

    # keep/drop function results
    if include_all_function_results:
        # keep all function results around as SYSTEM messages
        steps.append(MergeConsecutive(role=ChatRole.FUNCTION, joiner=function_result_joiner))
    elif include_last_function_result:
        # merge consecutive FUNCTION messages then remove all but the last (if it's the last message)

        def remover(m, ctx):
            return None if not ctx.is_last_of_type else m

        steps.append(MergeConsecutive(role=ChatRole.FUNCTION, joiner=function_result_joiner))
        steps.append(Apply(remover, role=ChatRole.FUNCTION))
    else:
        # remove all FUNCTION messages
        steps.append(Remove(role=ChatRole.FUNCTION))

    steps.append(
        ConversationFmt(
            prefix="<BOS_TOKEN>",
            generation_suffix=(
                f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{instruction_suffix}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
                if instruction_suffix
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

    return PromptPipeline(steps)


# ==== helpers ====
def function_prompt(f: AIFunction) -> str:
    """Build the Cohere python signature prompt for a given AIFunction."""
    params = f.get_params()

    # build params
    param_parts = []
    for param in params:
        param_parts.append(str(param))
    params_str = ", ".join(param_parts)

    # build docstring
    args = ""
    if len(params):
        args = f"\n\n    Args:\n        "
        doc_params = []
        for param in params:
            desc = f": {param.description}" if param.description else ""
            doc_params.append(f"{param.name} ({inspect.formatannotation(param.type)}){desc}")
        args += "\n        ".join(doc_params)

    # return
    return f'```python\ndef {f.name}({params_str}) -> List[Dict]:\n    """{f.desc}{args}\n    """\n    pass\n```'


CommandRToolCallInfo = namedtuple("CommandRToolCallInfo", "is_directly_answer filtered_tool_calls")


class CommandRMixin:
    """Common Command R functionality to share between engines"""

    def __init__(
        self,
        *args,
        tool_prompt_include_function_calls=True,
        tool_prompt_include_function_results=True,
        tool_prompt_instructions=DEFAULT_TOOL_INSTRUCTIONS,
        rag_prompt_include_function_calls=True,
        rag_prompt_include_function_results=True,
        rag_prompt_instructions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tool_prompt_include_function_calls = tool_prompt_include_function_calls

        self._default_pipeline = build_pipeline()
        self._tool_pipeline = build_pipeline(
            include_function_calls=tool_prompt_include_function_calls,
            include_all_function_results=tool_prompt_include_function_results,
            include_last_function_result=tool_prompt_include_function_results,
            instruction_suffix=tool_prompt_instructions,
        )
        self._rag_pipeline = build_pipeline(
            include_function_calls=rag_prompt_include_function_calls,
            include_all_function_results=rag_prompt_include_function_results,
            include_last_function_result=True,
            instruction_suffix=rag_prompt_instructions,
        )

    # ==== prompting ====
    def build_prompt(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None) -> str:
        # no functions: we can just do the default simple format
        if not functions:
            prompt = self._default_pipeline(messages)
            log.debug(f"PROMPT: {prompt}")
            return prompt

        # if we do have functions things get wacky
        # is the last message a FUNCTION? if so, we need to use the RAG template
        if messages and messages[-1].role == ChatRole.FUNCTION:
            prompt = self._build_prompt_rag(messages)
            log.debug(f"RAG PROMPT: {prompt}")
            return prompt

        # otherwise use the TOOL template
        prompt = self._build_prompt_tools(messages, functions)
        log.debug(f"TOOL PROMPT: {prompt}")
        return prompt

    def _build_prompt_tools(self, messages: list[ChatMessage], functions: list[AIFunction]):
        # get the function definitions
        function_text = "\n\n".join(map(function_prompt, functions))
        tool_prompt = DEFAULT_TOOL_PROMPT.format(user_functions=function_text)

        # wrap the initial system message, if any
        messages = messages.copy()
        if messages and messages[0].role == ChatRole.SYSTEM:
            messages[0] = messages[0].copy_with(content=DEFAULT_PREAMBLE + messages[0].text + tool_prompt)
        # otherwise add it in
        else:
            messages.insert(0, ChatMessage.system(DEFAULT_PREAMBLE + DEFAULT_TASK + tool_prompt))

        return self._tool_pipeline(messages)

    def _build_prompt_rag(self, messages: list[ChatMessage]):
        # wrap the initial system message, if any
        messages = messages.copy()
        if messages and messages[0].role == ChatRole.SYSTEM:
            messages[0] = messages[0].copy_with(content=DEFAULT_PREAMBLE + messages[0].text)
        # otherwise add it in
        else:
            messages.insert(0, ChatMessage.system(DEFAULT_PREAMBLE + DEFAULT_TASK))

        return self._rag_pipeline(messages)

    # ==== completions ====
    @staticmethod
    def _parse_completion(content: str, parse_functions=True, **kwargs) -> Completion:
        """Given the completion string, parse out any function calls."""
        log.debug(f"COMPLETION: {content}")

        # if we have tools, possibly parse out the Action
        tool_calls = None
        if parse_functions and (
            action_json := re.match(r"Action:\s*```json\n(.+)\n```", content, re.IGNORECASE | re.DOTALL)
        ):
            actions = json.loads(action_json.group(1))

            # translate back to kani spec
            tool_calls = []
            for action in actions:
                tool_name = action["tool_name"]
                tool_args = json.dumps(action["parameters"])
                tool_call = ToolCall.from_function_call(FunctionCall(name=tool_name, arguments=tool_args))
                tool_calls.append(tool_call)

            content = None
            log.debug(f"PARSED TOOL CALLS: {tool_calls}")

        return Completion(ChatMessage.assistant(content, tool_calls=tool_calls), **kwargs)

    @staticmethod
    def _toolcall_info(tool_calls: list[ToolCall]) -> CommandRToolCallInfo:
        """Return an info tuple containing Command R-specific metadata (is_directly_answer, filtered_tcs)."""
        tool_calls = tool_calls or []

        # if tool says directly answer, stream with the rag pipeline (but no result)
        if len(tool_calls) == 1 and tool_calls[0].function.name == "directly_answer":
            return CommandRToolCallInfo(is_directly_answer=True, filtered_tool_calls=[])
        # if the model generated multiple calls that happen to include a directly_answer, remove the directly_answer
        # then yield as normal
        tool_calls = [tc for tc in tool_calls if tc.function.name != "directly_answer"]
        return CommandRToolCallInfo(is_directly_answer=False, filtered_tool_calls=tool_calls)
