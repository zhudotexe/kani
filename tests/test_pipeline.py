import itertools

import pytest
from hypothesis import given, strategies as st

from kani import AIFunction, ChatMessage, ChatRole, FunctionCall, PromptPipeline, ToolCall
from kani.prompts.base import FilterMixin, PipelineStep
from kani.prompts.examples import ALL_EXAMPLE_KWARGS, build_conversation
from kani.prompts.types import PipelineMsgT


# ==== bases ====
class DummyFilterStep(FilterMixin, PipelineStep):
    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]):
        return list(self.filtered(msgs))


EXAMPLE_MSGS = list(itertools.chain.from_iterable(build_conversation(**{k: True for k in ALL_EXAMPLE_KWARGS})))


def test_filter_none():
    # no filter does nothing
    pipe = PromptPipeline([DummyFilterStep()])
    assert pipe(EXAMPLE_MSGS) == EXAMPLE_MSGS


@given(st.sampled_from(ChatRole))
def test_filter_role(role):
    # role filter
    pipe = PromptPipeline([DummyFilterStep(role=role)])
    assert all(m.role == role for m in pipe(EXAMPLE_MSGS))


@given(st.lists(st.sampled_from(ChatRole), min_size=1))
def test_filter_roles(roles):
    pipe = PromptPipeline([DummyFilterStep(role=roles)])
    assert all(m.role in roles for m in pipe(EXAMPLE_MSGS))


def test_filter_predicate():
    # only messages with a tool call
    pipe = PromptPipeline([DummyFilterStep(predicate=lambda msg: msg.tool_calls and len(msg.tool_calls) > 0)])
    assert all(m.tool_calls for m in pipe(EXAMPLE_MSGS))

    # only messages whose text is longer than 10 characters
    pipe = PromptPipeline([DummyFilterStep(predicate=lambda msg: msg.text and len(msg.text) > 10)])
    assert all(len(m.text) > 10 for m in pipe(EXAMPLE_MSGS) if m.text)


def test_filter_predicate_and_role():
    # only USER messages with a tool call (should be none)
    pipe = PromptPipeline(
        [DummyFilterStep(role=ChatRole.USER, predicate=lambda msg: msg.tool_calls and len(msg.tool_calls) > 0)]
    )
    assert len(pipe(EXAMPLE_MSGS)) == 0


# ==== steps ====
def test_translate_role():
    # should translate the role but nothing else
    pipe = PromptPipeline().translate_role(to=ChatRole.USER)
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert all(m.role == ChatRole.USER for m in pipe(EXAMPLE_MSGS))
    assert all(m.content == original.content for m, original in zip(pipe(EXAMPLE_MSGS), EXAMPLE_MSGS))


def test_wrap():
    # prefix
    pipe = PromptPipeline().wrap(prefix="prefix!")
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert all(m.text == f"prefix!{original.text or ''}" for m, original in zip(pipe(EXAMPLE_MSGS), EXAMPLE_MSGS))

    # suffix
    pipe = PromptPipeline().wrap(suffix="!suffix")
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert all(m.text == f"{original.text or ''}!suffix" for m, original in zip(pipe(EXAMPLE_MSGS), EXAMPLE_MSGS))

    # both
    pipe = PromptPipeline().wrap(prefix="prefix!", suffix="!suffix")
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert all(
        m.text == f"prefix!{original.text or ''}!suffix" for m, original in zip(pipe(EXAMPLE_MSGS), EXAMPLE_MSGS)
    )

    # neither
    pipe = PromptPipeline().wrap()
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert all(m.text == (original.text or "") for m, original in zip(pipe(EXAMPLE_MSGS), EXAMPLE_MSGS))


@given(st.sampled_from(ChatRole))
def test_merge_consecutive(merge_role):
    msgs = [
        ChatMessage.system("sys1"),
        ChatMessage.system("sys2"),
        ChatMessage.user("user1"),
        ChatMessage.user("user2"),
        ChatMessage.assistant("asst1"),
        ChatMessage.assistant("asst2"),
        ChatMessage.function("func", "func1"),
        ChatMessage.function("func", "func2"),
    ]

    # one role
    pipe = PromptPipeline().merge_consecutive(role=merge_role, sep="\n")
    assert len(pipe(msgs)) == len(msgs) - 1
    assert sum(1 for msg in pipe(msgs) if msg.role == merge_role) == 1
    assert sum(1 for msg in pipe(msgs) if msg.role != merge_role) == 6

    # all into one
    pipe = PromptPipeline().merge_consecutive(
        role=[ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT, ChatRole.FUNCTION], sep="\n", out_role=ChatRole.USER
    )
    assert len(pipe(msgs)) == 1
    assert pipe(msgs)[0].role == ChatRole.USER
    assert pipe(msgs)[0].text == "sys1\nsys2\nuser1\nuser2\nasst1\nasst2\nfunc1\nfunc2"

    with pytest.raises(ValueError):
        # missing out_role
        PromptPipeline().merge_consecutive(
            role=[ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT, ChatRole.FUNCTION], sep="\n"
        )


def test_function_call_fmt():
    pipe = PromptPipeline().function_call_fmt(
        lambda tc: "I'm a function call!", prefix="prefix!", suffix="!suffix", sep="\n===\n"
    )

    # one function call
    msgs = [
        ChatMessage.assistant(
            content=None, function_call=FunctionCall.with_args("get_weather", location="Tokyo, JP", unit="celsius")
        )
    ]
    assert pipe(msgs)[0].content == "prefix!I'm a function call!!suffix"

    # one with content
    msgs = [
        ChatMessage.assistant(
            content="content", function_call=FunctionCall.with_args("get_weather", location="Tokyo, JP", unit="celsius")
        )
    ]
    assert pipe(msgs)[0].content == "contentprefix!I'm a function call!!suffix"

    # many function calls
    msgs = [
        ChatMessage.assistant(
            content=None,
            tool_calls=[
                ToolCall.from_function("make_food", food="coffee", appliance="teapot"),
                ToolCall.from_function("make_food", food="cereal", dish="bowl"),
            ],
        ),
    ]
    assert pipe(msgs)[0].content == "prefix!I'm a function call!\n===\nI'm a function call!!suffix"

    # no function call
    msgs = [ChatMessage.assistant(content="Just passing by")]
    assert pipe(msgs) == msgs


def test_remove():
    # with no filters it removes everything
    # and we know the filters work, so...
    pipe = PromptPipeline().remove()
    assert len(pipe(EXAMPLE_MSGS)) == 0

    # if it doesn't match anything it shouldn't do anything
    pipe = PromptPipeline().remove(predicate=lambda _: False)
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert pipe(EXAMPLE_MSGS) == EXAMPLE_MSGS


@given(st.sampled_from(ChatRole))
def test_ensure_start(role):
    pipe = PromptPipeline().ensure_start(role=role)
    assert len(pipe(EXAMPLE_MSGS))
    assert pipe(EXAMPLE_MSGS)[0].role == role

    if EXAMPLE_MSGS[0].role == role:
        assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)


def test_apply():
    # noop apply
    pipe = PromptPipeline().apply(lambda m: m)
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert pipe(EXAMPLE_MSGS) == EXAMPLE_MSGS

    # remove everything
    pipe = PromptPipeline().apply(lambda m: None)
    assert len(pipe(EXAMPLE_MSGS)) == 0

    # 2 args: translate last to SYSTEM
    def translate_last(msg, ctx):
        if ctx.is_last:
            msg.role = ChatRole.SYSTEM
        return msg

    pipe = PromptPipeline().apply(translate_last)
    assert len(pipe(EXAMPLE_MSGS)) == len(EXAMPLE_MSGS)
    assert pipe(EXAMPLE_MSGS)[-1].role == ChatRole.SYSTEM

    # 3 args: translate even msgs to SYSTEM
    def translate_even(msg, ctx):
        if ctx.idx % 2 == 0:
            msg.role = ChatRole.SYSTEM
        return msg

    pipe = PromptPipeline().apply(translate_even)
    msgs = pipe(EXAMPLE_MSGS)
    assert len(msgs) == len(EXAMPLE_MSGS)
    for idx, (msg, original) in enumerate(zip(msgs, EXAMPLE_MSGS)):
        if idx % 2 == 0:
            assert msg.role == ChatRole.SYSTEM
        else:
            assert msg == original
