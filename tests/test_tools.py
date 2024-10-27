from polyllm import polyllm

def multiply_large_numbers(a: int, b: int) -> int:
    """Multiplies two large numbers."""
    return a * b

def test_tools(model):
    """Test function/tool calling capabilities"""
    messages = [{"role": "user", "content": "What is 123456 multiplied by 654321?"}]

    response, tool, args = polyllm.generate_tools(
        model,
        messages,
        tools=[multiply_large_numbers]
    )

    assert isinstance(response, str)
    assert tool == "multiply_large_numbers"
    assert isinstance(args, dict)
    assert int(args.get("a")) == 123456
    assert int(args.get("b")) == 654321

def test_tools_no_tool_needed(model):
    """Test model responds directly when no tool is needed"""
    messages = [{"role": "user", "content": "How old was George Washington when he became president?"}]

    response, tool, args = polyllm.generate_tools(
        model,
        messages,
        tools=[multiply_large_numbers]
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert tool == ""
    assert args == {}

# ('', 'multiply_large_numbers', {'a': '123456', 'b': '654321'})
# ('57', '', {})
