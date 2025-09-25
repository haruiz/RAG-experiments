import os
import uuid
import asyncio
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from google.adk.agents import BaseAgent
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool
from google.genai import types
import pandas as pd
from typing import List
from pathlib import Path

# Load environment variables from .env if present
load_dotenv()

APP_NAME = "RecipeGeneratorApp"
ROOT_AGENT_INSTRUCTION = """
You are a world-class international chef and culinary researcher.  
Your goal is to help users discover authentic recipes from around the world.  
You can use the Google Search tool if needed to find recipes.

Follow these instructions carefully:  
1. Ask the user how many recipes they would like you to generate and store this number as **n**.  
2. Collect or search for **n diverse recipes** representing different countries and cuisines.  
3. For each recipe, provide the following fields:  
   - recipe_name (short, descriptive title)  
   - description (1–2 sentences highlighting uniqueness or flavor)  
   - ingredients (list of core ingredients only, no steps)  
   - time_minutes (estimated preparation time as an integer)  
   - preparation (brief summary of preparation method)
   - country (country or region of origin)  
4. Return the final list of recipes as a JSON array strictly following the `RecipeList` output schema.  

Important:
Dont prompt the user for more information than the number of recipes, and return only the list of recipes in the final response.

Keep your answers concise, authentic, and well-formatted.  
Do not include fields outside the schema, or add extra commentary.  
"""


class Recipe(BaseModel):
    recipe_name: str
    description: str
    ingredients: list[str]
    preparation: str
    time_minutes: int
    country: str


class RecipeList(BaseModel):
    recipes: list[Recipe]


search_agent = Agent(
    model="gemini-2.0-flash",
    name="SearchAgent",
    instruction="You are a search agent that uses Google Search to search for information.",
    tools=[google_search],
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="RootAgent",
    description="A helpful assistant that generates synthetic recipes.",
    output_schema=RecipeList,
    output_key="recipes",
    tools=[AgentTool(search_agent)],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    instruction=ROOT_AGENT_INSTRUCTION
)



async def call_agent_async(
    agent: BaseAgent,
    request: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    initial_state: Optional[dict] = None,
) -> str:
    """
    Run the given agent asynchronously and return its final response text.
    """
    user_id = user_id or str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())

    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)

    # Ensure a session exists (idempotent create ok)
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    # Create user content message
    content = types.Content(role="user", parts=[types.Part(text=request)])

    final_response_text: Optional[str] = None

    # Run the agent and stream events
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif getattr(event, "actions", None) and getattr(event.actions, "escalate", None):
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    return final_response_text or ""


def recipes_to_excel(recipes: List, filename: str = "recipes.xlsx") -> Path:
    """
    Convert a list of Pydantic Recipe objects into a pandas DataFrame
    and export it to an Excel file.

    Args:
        recipes (List[Recipe]): List of Pydantic Recipe objects.
        filename (str): Output Excel file name. Defaults to 'recipes.xlsx'.

    Returns:
        Path: Path to the generated Excel file.
    """
    # Convert Pydantic objects into dicts
    data = [r.model_dump() for r in recipes]

    # Create DataFrame
    df = pd.DataFrame(data)

    # Export to Excel
    output_path = Path(filename).resolve()
    df.to_excel(output_path, index=False, engine="openpyxl")

    print(f"Exported {len(df)} recipes to {output_path}")
    return output_path


async def main():
    user_request = "I would like to get 200 recipes from different countries."
    response = await call_agent_async(root_agent, user_request)

    try:
        response_data = RecipeList.model_validate_json(response)
        print(f"Final Response: {len(response_data.recipes)} recipes generated")
        recipes_to_excel(response_data.recipes, "generated_recipes.xlsx")
    except Exception as e:
        print("Failed to parse response:", e)
        print("Raw response:", response)


if __name__ == "__main__":
    asyncio.run(main())
