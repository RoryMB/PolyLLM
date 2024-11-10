import json
import textwrap
from polyllm import polyllm

def test_json(model):
    """Test JSON output mode across all models"""
    messages = [
        {
            "role": "user",
            "content": textwrap.dedent("""
                Find the name of the first president of the USA and get the years that he served.
                Produce the result in JSON that matches this schema:
                    {
                        "first_name": "first name",
                        "last_name":  "last name",
                        "years_served": "years served"
                    }
            """).strip(),
        },
    ]

    correct = {
        "first_name": "George",
        "last_name": "Washington",
        "years_served": "1789-1797",
    }

    response = polyllm.generate(model, messages, json_output=True)
    assert isinstance(response, str)
    assert len(response) > 0

    data = json.loads(response)
    assert isinstance(data, dict)
    assert data == correct
