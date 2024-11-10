import textwrap
from pydantic import BaseModel, Field
from polyllm import polyllm

class Flight(BaseModel):
    departure_time: str = Field(description="The time the flight departs")
    destination: str = Field(description="The destination of the flight")

class FlightList(BaseModel):
    flights: list[Flight] = Field(description="A list of known flight details")

def test_structured(model):
    """Test structured output using Pydantic models"""
    flight_list_schema = polyllm.structured_output_model_to_schema(FlightList, indent=2)
    messages = [
        {
            "role": "user",
            "content": textwrap.dedent("""
                Write a list of 2 to 5 random flight details.
                Produce the result in JSON that matches this schema:
            """).strip() + "\n" + flight_list_schema,
        },
    ]

    response = polyllm.generate(model, messages, structured_output_model=FlightList)
    assert isinstance(response, str)
    assert len(response) > 0

    # Verify we can parse it into our Pydantic model
    polyllm.structured_output_to_object(response, FlightList)
