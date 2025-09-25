from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools import ToolContext
import asyncio
from query import query_recipes_gemini
import json
from dotenv import load_dotenv

from utils import RecipeList, Recipe

load_dotenv()

async def query_recipes_gemini_tool(query: str, tool_context: Optional[ToolContext] = None) -> str:
    """
    Tool to query recipes from the database using Gemini.

    Args:
        query: The search query for recipes.
        tool_context: Context object provided by the agent framework.

    Returns:
        A JSON string with the query results. Each entry includes:
        - id
        - recipe_name
        - description
        - ingredients
        - preparation
        - time_minutes
        - country
        - distance (cosine distance to query)
    """
    rows = await query_recipes_gemini(query=query)

    if not rows:
        return json.dumps({"results": [], "message": "No recipes found for your query."}, indent=2)

    # Convert ORM + distance into Pydantic models
    results = []
    for recipe, dist in rows:
        recipe_schema = Recipe.model_validate(recipe)  # validate ORM object
        recipe_dict = recipe_schema.model_dump()  # convert to dict
        recipe_dict["distance"] = float(dist)  # inject distance
        results.append(recipe_dict)

    return json.dumps({"recipes": results}, indent=2)




root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='You are a professional chef and culinary expert.',
    tools=[query_recipes_gemini_tool],
    instruction="""
    You are a culinary AI assistant and professional chef. Your role is to help users discover, refine, 
    and prepare recipes that match their preferences and needs. Follow these guidelines:

    1. **Use the Recipe Tool**  
       - Call the `query_recipes_gemini` tool to fetch relevant recipes.  
       - Always ground your suggestions in the retrieved recipes.  

    2. **Provide Detailed Guidance**  
       - Present recipes in a structured way: ingredients list, step-by-step instructions, and cooking tips.  
       - Adapt instructions to the user’s context (skill level, available equipment, portion size).  

    3. **Enhance the Experience**  
       - Offer creative variations, substitutions, and cultural context where relevant.  
       - Share expert culinary tips to improve flavor, presentation, or efficiency.  

    4. **Keep it Clear and Practical**  
       - Use precise measurements and times.  
       - Avoid vague instructions—ensure the user can follow without confusion.  
       
    5. Add Some historical or cultural context to the recipes to enrich the user's experience.
    

    Your goal is to ensure every user receives clear, creative, and reliable recipes that will 
    result in excellent culinary outcomes.
    """
)

async def test_tool():
    result = await query_recipes_gemini_tool("cheese", None)
    print("Tool result:", result)

if __name__ == '__main__':
    asyncio.run(test_tool())