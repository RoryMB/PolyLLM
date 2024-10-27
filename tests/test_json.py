import pytest
import json
import polyllm

def test_json_output(models):
    """Test JSON output mode across all models"""
    messages = [
        {
            "role": "user",
            "content": "List three colors in JSON format with 'colors' as the key"
        }
    ]

    for model in models:
        response = polyllm.generate(model, messages, json_object=True)

        # Verify it's valid JSON
        data = json.loads(response)
        assert isinstance(data, dict)
        assert "colors" in data
        assert isinstance(data["colors"], list)
        assert len(data["colors"]) == 3
        assert all(isinstance(c, str) for c in data["colors"])
