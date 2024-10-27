import pytest
from pydantic import BaseModel, Field
import polyllm

class Flight(BaseModel):
    departure_time: str = Field(description="The time the flight departs")
    destination: str = Field(description="The destination of the flight")

class FlightList(BaseModel):
    flights: list[Flight] = Field(description="A list of known flight details")

def test_structured_output(models):
    """Test structured output using Pydantic models"""
    messages = [
        {
            "role": "user",
            "content": "Write a list of 2 to 5 random flight details."
        }
    ]

    for model in models:
        # Skip models that don't support structured output
        if isinstance(model, str) and (model.startswith("ollama/") or model in polyllm.anthropic_models):
            continue

        response = polyllm.generate(model, messages, json_schema=FlightList)

        # Verify we can parse it into our Pydantic model
        result = polyllm.json_to_pydantic(response, FlightList)
        assert isinstance(result, FlightList)
        assert len(result.flights) >= 2
        assert len(result.flights) <= 5

        for flight in result.flights:
            assert isinstance(flight.departure_time, str)
            assert isinstance(flight.destination, str)
