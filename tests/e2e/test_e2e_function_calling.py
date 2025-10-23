"""
E2E tests for function calling capabilities.

Tests cover:
- Single-round function calling (one function call, then response)
- Multi-round function calling (multiple function calls across turns)
- Parallel function calling (multiple functions in one turn)
- Error handling and retry
- Different after behaviors (model vs user turn)
"""

import enum
from typing import Annotated

import pytest
from pytest_lazy_fixtures import lf

from kani import AIParam, ChatRole, Kani, ai_function, print_stream, print_width
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.freeze_uuids(side_effect="random"),
]


# ==== Test Kani Classes ====
class CalculatorKani(Kani):
    """Simple calculator for testing single-round function calling."""

    @ai_function()
    def add(self, a: int, b: int):
        """Add two numbers together."""
        return a + b

    @ai_function()
    def multiply(self, a: int, b: int):
        """Multiply two numbers together."""
        return a * b

    @ai_function()
    def divide(self, a: float, b: float):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero!")
        return a / b


class Unit(enum.Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


class WeatherKani(Kani):
    """Weather API for testing parallel function calling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weather_data = {
            "San Francisco, CA": {"temp_f": 65, "temp_c": 18, "conditions": "Foggy"},
            "New York, NY": {"temp_f": 75, "temp_c": 24, "conditions": "Sunny"},
            "Seattle, WA": {"temp_f": 55, "temp_c": 13, "conditions": "Rainy"},
            "Miami, FL": {"temp_f": 85, "temp_c": 29, "conditions": "Humid"},
        }

    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
        unit: Unit,
    ):
        """Get the current weather in a given location."""
        data = self.weather_data.get(location)
        if not data:
            return f"Weather data not available for {location}."
        temp = data["temp_f"] if unit == Unit.FAHRENHEIT else data["temp_c"]
        return f"Weather in {location}: {data['conditions']}, {temp} degrees {unit.value}."

    @ai_function()
    def get_forecast(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
    ):
        """Get the 3-day forecast for a location."""
        if location not in self.weather_data:
            return f"Forecast not available for {location}."
        return f"3-day forecast for {location}: Day 1: Partly cloudy, Day 2: Sunny, Day 3: Rainy"


class DatabaseKani(Kani):
    """Mock database for testing multi-round function calling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database = {
            "users": [
                {"id": 1, "name": "Alice", "age": 30, "city": "San Francisco"},
                {"id": 2, "name": "Bob", "age": 25, "city": "New York"},
                {"id": 3, "name": "Charlie", "age": 35, "city": "Seattle"},
            ],
            "orders": [
                {"order_id": 101, "user_id": 1, "item": "Laptop", "price": 1200},
                {"order_id": 102, "user_id": 1, "item": "Mouse", "price": 25},
                {"order_id": 103, "user_id": 2, "item": "Keyboard", "price": 80},
                {"order_id": 104, "user_id": 3, "item": "Monitor", "price": 300},
            ],
        }

    @ai_function()
    def search_users(self, city: str):
        """Search for users by city."""
        results = [u for u in self.database["users"] if u["city"].lower() == city.lower()]
        if not results:
            return f"No users found in {city}."
        return str([{"id": u["id"], "name": u["name"]} for u in results])

    @ai_function()
    def get_user_orders(self, user_id: int):
        """Get all orders for a specific user ID."""
        orders = [o for o in self.database["orders"] if o["user_id"] == user_id]
        if not orders:
            return f"No orders found for user {user_id}."
        return str(orders)

    @ai_function()
    def get_order_total(self, user_id: int):
        """Calculate the total amount spent by a user."""
        orders = [o for o in self.database["orders"] if o["user_id"] == user_id]
        total = sum(o["price"] for o in orders)
        return total


class UserControlKani(Kani):
    """Test functions with after=ChatRole.USER."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notifications = []

    @ai_function(after=ChatRole.USER)
    def send_email(self, recipient: str, subject: str, body: str):
        """Send an email to a recipient. Control returns to the user after this function."""
        self.notifications.append(f"Email sent to {recipient}: {subject}")
        return f"Email sent to {recipient} with subject '{subject}'"

    @ai_function()
    def check_inbox(self):
        """Check the email inbox."""
        return "You have 3 new messages: 1 from boss, 1 from friend, 1 from newsletter"


class ErrorKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    @ai_function()
    def flaky_function(self, x: int):
        """A function that fails the first time."""
        self.call_count += 1
        if self.call_count == 1:
            raise ValueError("First call always fails!")
        return f"Success! x = {x}"


class CustomExceptionPromptKani(Kani):
    """Test custom exception prompt handling (from examples/3_customization_exception_prompt.py)."""

    async def handle_function_call_exception(self, call, err, attempt, *args, **kwargs):
        """Override error handling to use a custom system message."""
        # Get the standard retry logic...
        result = await super().handle_function_call_exception(call, err, attempt, *args, **kwargs)
        # But override the returned message with our own custom format
        result.message = result.message.copy_with(
            text=(
                f"CUSTOM ERROR HANDLER: The function {call.name} failed with error: {err}. Please inform the user about"
                " this error."
            ),
        )
        return result

    @ai_function()
    def get_time(self):
        """Get the current time in the user's time zone."""
        raise RuntimeError("The time API is currently offline (error 0xDEADBEEF).")

    @ai_function()
    def working_function(self):
        """A function that works correctly."""
        return "This function works!"


# ==== Helper Functions ====
async def _do_inference(ai, query, stream, **kwargs):
    """Helper to run inference with consistent output."""
    print_width(query, prefix="USER: ")
    if stream:
        async for strm in ai.full_round_stream(query, **kwargs):
            # assistant
            if strm.role == ChatRole.ASSISTANT:
                await print_stream(strm, prefix="AI: ")
                msg = await strm.message()
                text = assistant_message_thinking(msg, show_args=True)
                if text:
                    print_width(text, prefix="AI: ")
            # function
            elif strm.role == ChatRole.FUNCTION:
                msg = await strm.message()
                print_width(f"[{msg.name or 'FUNCTION'}] {msg.text}", prefix="FUNC: ")
        return ai.chat_history
    else:
        messages = []
        async for msg in ai.full_round(query, **kwargs):
            messages.append(msg)
            # assistant
            if msg.role == ChatRole.ASSISTANT:
                text = assistant_message_contents_thinking(msg, show_args=True, show_reasoning=True)
                if text:
                    print_width(text, prefix="AI: ")
            # function
            elif msg.role == ChatRole.FUNCTION:
                print_width(f"[{msg.name or 'FUNCTION'}] {msg.text}", prefix="FUNC: ")
        return messages


# ==== Tests ====
@pytest.mark.parametrize(
    "engine",
    [
        lf("e2e_anthropic_engine"),
        lf("e2e_google_engine"),
        lf("e2e_openai_engine"),
        lf("e2e_huggingface_engine"),
        lf("e2e_llamacpp_engine"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.request_model_capabilities(["function_calling"])
class TestE2EFunctionCalling:
    """E2E tests for function calling across all engines."""

    async def test_single_function_call(self, engine, stream):
        """Test basic function calling: one function call, then model responds."""
        ai = CalculatorKani(
            engine,
            system_prompt=(
                "You are a helpful calculator assistant. Use the provided functions to answer math questions."
            ),
        )
        messages = await _do_inference(ai, "What is 15 plus 27?", stream)

        # Should have at least: assistant (tool call) -> function result -> assistant response
        assert len(messages) >= 3
        # Check that a function was called
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 1
        # Check that the result is correct
        assert "42" in messages[-1].text

    async def test_multiple_operations(self, engine, stream):
        """Test multiple function calls in sequence."""
        ai = CalculatorKani(
            engine,
            system_prompt=(
                "You are a helpful calculator assistant. Use the provided functions to answer math questions."
            ),
        )
        messages = await _do_inference(ai, "What is (8 times 5) plus 10?", stream)

        # Should call multiply and add functions
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 2
        # Result should be 50
        assert "50" in messages[-1].text

    async def test_parallel_function_calls(self, engine, stream):
        """Test calling multiple functions in parallel (same turn)."""
        ai = WeatherKani(
            engine,
            system_prompt="You are a weather assistant. Use the provided functions to answer weather questions.",
        )
        messages = await _do_inference(
            ai, "What's the weather in San Francisco and New York? Report in Fahrenheit.", stream
        )

        # Should call get_weather at least twice (could be parallel or sequential)
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 2

        # Should mention both cities in the final response
        final_text = messages[-1].text.lower()
        assert "san francisco" in final_text or "francisco" in final_text
        assert "new york" in final_text or "york" in final_text

    async def test_multi_round_function_calling(self, engine, stream):
        """Test multi-round function calling where one function informs the next."""
        ai = DatabaseKani(
            engine,
            system_prompt=(
                "You are a database assistant. Use the provided functions to answer questions about users and orders."
            ),
        )
        messages = await _do_inference(
            ai, "Find all users in San Francisco and tell me their names and total order amounts.", stream
        )

        # Should call search_users, then get_order_total or get_user_orders
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 2

        # Should mention Alice (user in SF) and amounts
        final_text = messages[-1].text.lower()
        assert "alice" in final_text
        # Should have some number in response (order total)
        assert any(char.isdigit() for char in final_text)

    async def test_function_after_user(self, engine, stream):
        """Test functions with after=ChatRole.USER (returns control to user)."""
        ai = UserControlKani(
            engine,
            system_prompt="You are an email assistant. Use the provided functions to help with email tasks.",
        )
        messages = await _do_inference(
            ai, 'Send an email to john@example.com with subject "Meeting Tomorrow" and body "See you at 3pm"', stream
        )

        # Should call send_email function
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 1

        # The function should have been called
        assert len(ai.notifications) == 1
        assert "john@example.com" in ai.notifications[0]

        # Check that send_email was called
        send_email_calls = [m for m in function_calls if m.name == "send_email"]
        assert len(send_email_calls) >= 1

        # the last message should be from the function returning
        assert messages[-1].role == ChatRole.FUNCTION

    async def test_combined_functions(self, engine, stream):
        """Test using different types of functions together."""
        ai = WeatherKani(
            engine,
            system_prompt="You are a weather assistant. Use the provided functions to answer weather questions.",
        )
        messages = await _do_inference(
            ai, "What's the current weather in Miami and what's the forecast for the next 3 days?", stream
        )

        # Should call both get_weather and get_forecast
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 2

        # Should have called different functions
        function_names = {m.name for m in function_calls}
        assert len(function_names) >= 2

        # Final response should mention both current weather and forecast
        final_text = messages[-1].text.lower()
        assert "miami" in final_text


@pytest.mark.parametrize(
    "engine",
    [
        lf("e2e_anthropic_engine"),
        lf("e2e_google_engine"),
        lf("e2e_openai_engine"),
        lf("e2e_huggingface_engine"),
        lf("e2e_llamacpp_engine"),
    ],
)
@pytest.mark.request_model_capabilities(["function_calling"])
class TestE2EFunctionCallingRetry:
    """Test retry behavior and max_function_rounds."""

    async def test_max_function_rounds(self, engine):
        """Test that max_function_rounds limits function calling rounds."""
        ai = CalculatorKani(
            engine,
            system_prompt="You are a calculator assistant. Use functions for every calculation.",
        )

        messages = await _do_inference(
            ai, "Calculate: (2+3) * (4+5) * (6+7). Use functions for each step.", stream=False, max_function_rounds=1
        )

        # With max_function_rounds=1, should only have one function calling round
        # i.e., only 1 asst message before the last that has functions
        assert len([m for m in messages[:-1] if m.role == ChatRole.ASSISTANT and m.tool_calls]) <= 1

    async def test_retry_on_error(self, engine):
        """Test that errors trigger retry behavior."""
        ai = ErrorKani(engine, retry_attempts=2)
        await _do_inference(
            ai, "Call the flaky_function with x=42. Retry if it fails, it should eventually work.", False
        )

        assert ai.call_count == 2

    async def test_custom_exception_prompt(self, engine):
        """Test custom exception prompt handling (from examples/3_customization_exception_prompt.py)."""
        ai = CustomExceptionPromptKani(
            engine,
            system_prompt="You are a helpful assistant. Use the provided functions to answer questions.",
        )

        messages = await _do_inference(ai, "What time is it?", False)

        # Should have a function response that's and error
        function_calls = [m for m in messages if m.role == ChatRole.FUNCTION]
        assert len(function_calls) >= 1
        assert all(fc.is_tool_call_error for fc in function_calls)

        # The system message should have our custom error handler marker
        assert any("CUSTOM ERROR HANDLER" in m.text for m in function_calls)

        # Should mention the error in system message
        error_text = function_calls[0].text
        assert "get_time" in error_text
        assert "offline" in error_text or "0xDEADBEEF" in error_text
