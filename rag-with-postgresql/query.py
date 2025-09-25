import asyncio
import functools
import logging
import os
import time
from typing import Any, Sequence
from typing import Optional

import kagglehub
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, Row

from constants import GEMMA_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL, EMBED_DIM
from rag import RecipeORM, get_session
from utils import make_plot_figure

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from a .env file
load_dotenv(dotenv_path=".env")


def timeit(func):
    """
    A decorator that measures and logs the execution time of an async function.

    Args:
        func: The async function to decorate.

    Raises:
        TypeError: If the decorated function is not asynchronous.

    Returns:
        The wrapped async function with execution timing.
    """
    if not asyncio.iscoroutinefunction(func):
        raise TypeError("timeit decorator can only be used with async functions")

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


@timeit
async def query_recipes_gemma(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7
) -> Sequence[Row[tuple[RecipeORM, Any]]]:
    """
    Query the database for recipes similar to a text query using **Gemma embeddings**.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
        similarity_threshold: Cosine distance threshold. Lower = more similar.

    Returns:
        A list of rows where each row is a tuple of:
            (RecipeORM object, cosine distance score).
    """
    model_id = kagglehub.model_download(GEMMA_EMBEDDING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id).to(device=device)

    async with get_session() as session:
        query_embedding = model.encode(
            query,
            prompt_name="Retrieval-query"
        )
        select_expr = RecipeORM.gemma_embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(RecipeORM, select_expr)
            .where(select_expr < similarity_threshold)
            .order_by(select_expr.asc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        rows = result.all()

        for recipe, dist in rows:
            logger.info(
                f"Recipe: {recipe.recipe_name}, Country: {recipe.country}, "
                f"Time: {recipe.time_minutes} mins, Distance: {dist:.4f}"
            )

        return rows


@timeit
async def query_recipes_gemini(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7,
    client: Optional[genai.Client] = None
) -> Sequence[Row[tuple[RecipeORM, Any]]]:
    """
    Query the database for recipes similar to a text query using **Gemini embeddings**.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
        similarity_threshold: Cosine distance threshold. Lower = more similar.
        client: Optional genai.Client. If None, a new one will be created.

    Returns:
        A list of rows where each row is a tuple of:
            (RecipeORM object, cosine distance score).
    """
    if client is None:
        client = genai.Client()

    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[types.Part.from_text(text=query)],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM
        )
    )

    async with get_session() as session:
        query_embedding = np.array(result.embeddings[0].values)
        normed_embedding = query_embedding / np.linalg.norm(query_embedding)
        logger.debug(f"Normed embedding length: {len(normed_embedding)}")
        logger.debug(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}")

        select_expr = RecipeORM.gemini_embedding.cosine_distance(normed_embedding).label("distance")
        stmt = (
            select(RecipeORM, select_expr)
            .where(select_expr < similarity_threshold)
            .order_by(select_expr.asc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        rows = result.all()

        for recipe, dist in rows:
            logger.info(
                f"Recipe: {recipe.recipe_name}, Country: {recipe.country}, "
                f"Time: {recipe.time_minutes} mins, Distance: {dist:.4f}"
            )

        return rows


async def test_rag(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7
) -> None:
    """
    Run a RAG query against both Gemma and Gemini embeddings and print results.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
        similarity_threshold: Cosine distance threshold for filtering.
    """
    await query_recipes_gemini(query, limit, similarity_threshold)
    await query_recipes_gemma(query, limit, similarity_threshold)


async def query_plot_data(
    embeddings_field: str = "gemini_embedding"
) -> Sequence[Row]:
    """
    Fetch recipe embeddings and metadata from the database.

    Args:
        embeddings_field: Either 'gemini_embedding' or 'gemma_embedding'.

    Raises:
        ValueError: If an invalid field is provided.
        RuntimeError: If no rows are returned.

    Returns:
        A list of rows, each containing:
            (embedding vector, country, recipe_name).
    """
    if embeddings_field not in {"gemini_embedding", "gemma_embedding"}:
        raise ValueError("embeddings_field must be 'gemini_embedding' or 'gemma_embedding'")

    async with get_session() as session:
        query = select(
            getattr(RecipeORM, embeddings_field),
            RecipeORM.country,
            RecipeORM.recipe_name
        )
        result = await session.execute(query)
        rows = result.all()

    if not rows:
        raise RuntimeError("No rows found.")

    return rows


async def plot_embeddings(
    method: str = "tsne",
    embeddings_field: str = "gemini_embedding"
) -> None:
    """
    Generate and display a 2D scatter plot of recipe embeddings.

    Args:
        method: Dimensionality reduction method ('tsne' or 'pca').
        embeddings_field: Embedding field to use ('gemini_embedding' or 'gemma_embedding').

    Returns:
        None. Displays the plot and optionally saves it.
    """
    rows = await query_plot_data(embeddings_field)
    make_plot_figure(rows, method=method)


if __name__ == '__main__':
    asyncio.run(test_rag("Find quick South America dishes with corn"))
    #asyncio.run(plot_embeddings(method="tsne", embeddings_field="gemini_embedding"))
