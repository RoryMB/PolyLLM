import pytest
import polyllm

def multiply(x: int, y: int) -> int:
    """Multiply two numbers"""
    return x * y

def test_tool_usage(models):
    """Test function/tool calling capabilities"""
    messages = [
        {
            "role": "user",
            "content": "What is 7 times 6?"
        }
    ]

    for model in models:
        response, tool, args = polyllm.generate_tools(
            model,
            messages,
            tools=[multiply]
        )

        # Should use the multiply tool
        assert tool == "multiply"
        assert args == {"x": 7, "y": 6}

def test_tool_usage_no_tool_needed(models):
    """Test model responds directly when no tool is needed"""
    messages = [
        {
            "role": "user",
            "content": "Say hello!"
        }
    ]

    for model in models:
        response, tool, args = polyllm.generate_tools(
            model,
            messages,
            tools=[multiply]
        )

        # Should not use any tool
        assert tool == ""
        assert args == {}
        assert isinstance(response, str)
        assert "hello" in response.lower()
