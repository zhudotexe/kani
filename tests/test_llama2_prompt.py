"""Tests to ensure the LLaMA v2 prompt is correct."""

from kani import ChatMessage
from kani.prompts.impl import LLAMA2_PIPELINE


def prompt_str(messages: list[ChatMessage]) -> str:
    return LLAMA2_PIPELINE(messages)


def test_basic():
    messages = [
        ChatMessage.user("Hello there."),
        ChatMessage.assistant("General Kenobi."),
    ]
    expected = "<s>[INST] Hello there. [/INST] General Kenobi."
    assert prompt_str(messages) == expected

    messages = [
        ChatMessage.system("I am a system message."),
        ChatMessage.user("Hello there."),
        ChatMessage.assistant("General Kenobi."),
    ]
    expected = "<s>[INST] <<SYS>>\nI am a system message.\n<</SYS>>\n\nHello there. [/INST] General Kenobi."
    assert prompt_str(messages) == expected


def test_2round():
    messages = [
        ChatMessage.system("I am a system message."),
        ChatMessage.user("Hello there."),
        ChatMessage.assistant("General Kenobi."),
        ChatMessage.user("I am:"),
        ChatMessage.assistant("a potato."),
    ]
    expected = (
        "<s>[INST] <<SYS>>\nI am a system message.\n<</SYS>>\n\nHello there. [/INST] General Kenobi. </s>"
        "<s>[INST] I am: [/INST] a potato."
    )
    assert prompt_str(messages) == expected


def test_2system():
    messages = [
        ChatMessage.system("I am a system message."),
        ChatMessage.system("But wait, there's more."),
        ChatMessage.user("Hello there."),
        ChatMessage.assistant("General Kenobi."),
    ]
    expected = (
        "<s>[INST] <<SYS>>\nI am a system message.\n<</SYS>>\n\n"
        "<<SYS>>\nBut wait, there's more.\n<</SYS>>\n\nHello there. [/INST] General Kenobi."
    )
    assert prompt_str(messages) == expected


def test_2user():
    messages = [
        ChatMessage.system("I am a system message."),
        ChatMessage.user("Hello there."),
        ChatMessage.user("I am another message."),
        ChatMessage.assistant("General Kenobi."),
    ]
    expected = (
        "<s>[INST] <<SYS>>\nI am a system message.\n<</SYS>>\n\n"
        "Hello there.\nI am another message. [/INST] General Kenobi."
    )
    assert prompt_str(messages) == expected


def test_2asst():
    messages = [
        ChatMessage.system("I am a system message."),
        ChatMessage.user("Hello there."),
        ChatMessage.assistant("General Kenobi."),
        ChatMessage.assistant("I am another message."),
    ]
    expected = (
        "<s>[INST] <<SYS>>\nI am a system message.\n<</SYS>>\n\n"
        "Hello there. [/INST] General Kenobi. I am another message."
    )
    assert prompt_str(messages) == expected
