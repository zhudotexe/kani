import inspect
from typing import Annotated

from kani.ai_function import AIFunction, AIParam
from kani.models import ChatMessage, FunctionCall, ToolCall


def build_conversation(
    *,
    function_call=False,
    consecutive_user=False,
    consecutive_assistant=False,
    consecutive_system=False,
    multi_function_call=False,
    end_on_assistant=False,
) -> list[list[ChatMessage]]:
    """Return a list of groups of messages for a test case"""
    function_call_msgs = []
    consecutive_user_msgs = []
    consecutive_assistant_msgs = []
    consecutive_system_msgs = []
    multi_function_call_msgs = []
    end_on_assistant_msgs = []

    if function_call:
        # with and without content
        function_call_msgs = [
            ChatMessage.user("What's the weather in Tokyo?"),
            ChatMessage.assistant(
                content=None, function_call=FunctionCall.with_args("get_weather", location="Tokyo, JP", unit="celsius")
            ),
            ChatMessage.function("get_weather", "Weather in Tokyo, JP: Partly cloudy, 21 degrees celsius."),
            ChatMessage.assistant("It's partly cloudy and 21 degrees in Tokyo."),
            ChatMessage.user("In Fahrenheit please."),
            ChatMessage.assistant(
                content="Let me check that.",
                function_call=FunctionCall.with_args("get_weather", location="Tokyo, JP", unit="fahrenheit"),
            ),
            ChatMessage.function("get_weather", "Weather in Tokyo, JP: Partly cloudy, 70 degrees fahrenheit."),
            ChatMessage.assistant("It's partly cloudy and 70 degrees in Tokyo."),
        ]

    if consecutive_user:
        consecutive_user_msgs = [
            ChatMessage.user("What does kani mean?"),
            ChatMessage.user("It's in Japanese."),
            ChatMessage.assistant("Kani means 'crab' in Japanese."),
        ]

    if consecutive_assistant:
        consecutive_assistant_msgs = [
            ChatMessage.user("Ctrl-C!"),
            ChatMessage.assistant("Hey, don't"),
            ChatMessage.assistant("interrupt me!"),
        ]

    if consecutive_system:
        consecutive_system_msgs = [ChatMessage.system("Please follow the following instructions:")]

    if multi_function_call:
        multi_function_call_msgs = [
            ChatMessage.user("Please make me coffee and a bowl of cereal."),
            ChatMessage.assistant(
                content=None,
                tool_calls=[
                    tc := ToolCall.from_function("make_food", food="coffee", appliance="teapot"),
                    tc2 := ToolCall.from_function("make_food", food="cereal", dish="bowl"),
                ],
            ),
            ChatMessage.function("make_food", "Error 428: I'm a teapot!", tc.id),
            ChatMessage.function("make_food", "Beep boop. Cerealbot has been dispatched.", tc2.id),
            ChatMessage.assistant("You need a coffee pot to make coffee, but Cerealbot is on the way."),
        ]

    if end_on_assistant:
        end_on_assistant_msgs = [ChatMessage.assistant("African or European?")]

    grps = [
        [
            *consecutive_system_msgs,
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("Hello there."),
            ChatMessage.assistant("Hi! How can I help?"),
        ],
        function_call_msgs,
        multi_function_call_msgs,
        consecutive_user_msgs,
        consecutive_assistant_msgs,
        [
            ChatMessage.user("What is the airspeed velocity of an unladen swallow?"),
            *end_on_assistant_msgs,
        ],
    ]
    return [grp for grp in grps if grp]


def build_functions(*, function_call=False, **_) -> list[AIFunction]:
    """Return a list of AIFunctions for a test case"""

    # noinspection PyUnusedLocal
    def get_weather(location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")]):
        """Get the current weather in a given location."""
        pass

    fcs = []
    if function_call:
        fcs.append(AIFunction(get_weather))
    return fcs


ALL_EXAMPLE_KWARGS = list(inspect.signature(build_conversation).parameters.keys())
