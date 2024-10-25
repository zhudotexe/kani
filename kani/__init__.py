from . import engines, exceptions, utils
from ._version import __version__
from .ai_function import AIFunction, AIParam, ai_function
from .internal import ExceptionHandleResult, FunctionCallResult
from .kani import Kani
from .models import ChatMessage, ChatRole, FunctionCall, MessagePart, ToolCall
from .prompts import PromptPipeline
from .utils.cli import chat_in_terminal, chat_in_terminal_async, format_stream, format_width, print_stream, print_width

# declare that kani is also a namespace package
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
