from contextlib import asynccontextmanager

from pydantic import BaseModel
from sqlalchemy import URL
from pydantic_settings import BaseSettings
from sqlalchemy import Column, Integer, String, insert
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import registry
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector
import pandas as pd
import logging
import os
from typing import AsyncGenerator, Any, Optional
from dotenv import load_dotenv
from google import genai
import asyncio
from utils import _as_str, get_gemini_embeddings, get_gemma_embeddings, create_synthetic_dataset, Recipe
from constants import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from a .env file
load_dotenv(dotenv_path=".env")
# Initialize the GenAI client
client = genai.Client()



# Database settings using Pydantic
class DbSettings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "recipesdb"
    DB_USER: str = "<dbuser>"
    DB_PASSWORD: str = "<dbpassword>"

db_settings = DbSettings()
db_url = URL.create(
    drivername="postgresql+asyncpg",
    username=db_settings.DB_USER,
    password=db_settings.DB_PASSWORD,  # URL.create safely quotes special chars
    host=db_settings.DB_HOST,
    port=db_settings.DB_PORT,
    database=db_settings.DB_NAME,
)
engine = create_async_engine(db_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
mapper_registry = registry()

class Base(DeclarativeBase):
    registry = mapper_registry

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession | Any, Any]:
    async with SessionLocal() as session:
        yield session


class RecipeORM(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recipe_name = Column(String, index=True)
    description = Column(String)
    ingredients = Column(String)  # Store as a comma-separated string
    preparation = Column(String)
    time_minutes = Column(Integer)
    country = Column(String)
    gemini_embedding = Column(Vector(EMBED_DIM))
    gemma_embedding = Column(Vector(EMBED_DIM))



async def ensure_database():
    """Creates the target database if it doesn't exist by connecting via the 'postgres' DB."""
    # connect to the admin DB to create the target DB
    admin_url = db_url.set(database="postgres")
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    try:
        async with admin_engine.begin() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_settings.DB_NAME},
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{db_settings.DB_NAME}"'))
                logger.info(f'Created database "{db_settings.DB_NAME}".')
    finally:
        await admin_engine.dispose()

async def ensure_extensions():
    """Installs required PostgreSQL extensions (e.g., 'vector') in the target database."""
    async with engine.begin() as conn:
        # Install pgvector extension; requires superuser or appropriate privs
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("Ensured extension: vector")

async def create_tables(perform_drop: bool = False):
    """Drops and recreates the database tables based on the SQLAlchemy ORM models.
    Args:
        perform_drop: If True, drops existing tables before creating them.

    """
    async with engine.begin() as conn:
        if perform_drop:
            await conn.run_sync(mapper_registry.metadata.drop_all)
        await conn.run_sync(mapper_registry.metadata.create_all)


def chunked(iterable: list, n: int) -> list:
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

async def populate_db(chunk_size: int = 2000):
    """Populates the database with recipes and their embeddings from the Parquet file.

    Args:
        chunk_size: The number of rows to insert in each bulk operation.
    """
    df = pd.read_parquet(EMBEDDING_DATASET_FILE, engine="pyarrow")

    # Build payload rows
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "recipe_name":      r.get("recipe_name"),
            "description":      r.get("description"),
            "ingredients":      ",".join(r.get("ingredients").strip("[]").replace("'", "").split(",")),
            "time_minutes":     r.get("time_minutes"),
            "preparation":      r.get("preparation"),
            "country":          r.get("country"),
            "gemini_embedding": r.get("gemini_embedding"),  # list[float] — pgvector will handle it
            "gemma_embedding":  r.get("gemma_embedding"),   # list[float] — pgvector will handle it
        })
    # Fast bulk insert in chunks
    async with SessionLocal() as session:
        stmt = insert(RecipeORM)
        for chunk in chunked(rows, chunk_size):
            async with session.begin():
                await session.execute(stmt, chunk)


def create_embeddings(limit: Optional[int] = None) -> None:
    """Reads the synthetic dataset, generates embeddings, and saves the result to a Parquet file.

    This function reads the `recipes.csv` file, constructs a unified text field for each
    recipe, and then generates embeddings using both Gemini and Gemma models. The resulting
    DataFrame, including the new embedding columns, is saved to a Parquet file.

    Args:
        limit: An optional integer to limit the number of rows processed from the
               input CSV. Useful for testing.
    """

    df = pd.read_csv(SYNTHETIC_DATASET_FILE)
    if limit:
        df = df.head(limit).copy()

    # Build the unified text field
    df["text"] = df.apply(
        lambda row: (
            f"Recipe: {_as_str(row.get('recipe_name'))}.\n"
            f"Description: {_as_str(row.get('description'))}.\n"
            f"Ingredients: {_as_str(row.get('ingredients'))}.\n"
            f"Country: {_as_str(row.get('country'))}.\n"
            f"Time to prepare: {_as_str(row.get('time_minutes'))} minutes."
        ),
        axis=1
    )

    texts = df["text"].tolist()
    # Compute embeddings only if missing
    if "gemini_embedding" not in df.columns or df["gemini_embedding"].isna().any():
        df["gemini_embedding"] = get_gemini_embeddings(texts, model_variant=GEMINI_EMBEDDING_MODEL, output_dim=EMBED_DIM, client=client)
    if "gemma_embedding" not in df.columns or df["gemma_embedding"].isna().any():
        df["gemma_embedding"] = get_gemma_embeddings(texts, model_variant=GEMMA_EMBEDDING_MODEL)

    df.to_parquet(EMBEDDING_DATASET_FILE, index=False, engine="pyarrow")
    logger.info(f"Embeddings created and saved to {EMBEDDING_DATASET_FILE} with {len(df)} records.")


async def create_indexes():
    """Creates IVFFlat indexes on the embedding columns for faster similarity searches."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gemini_embedding_cosine ON recipes USING ivfflat (gemini_embedding vector_cosine_ops) WITH (lists = 100)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gemma_embedding_cosine ON recipes USING ivfflat (gemma_embedding vector_cosine_ops) WITH (lists = 100)"))
        logger.info("Created indexes on embeddings.")


async def setup() -> None:
    """Runs the full data pipeline: create dataset, embeddings, and set up the database."""
    logger.info("Starting setup...")
    create_synthetic_dataset(client=client,
                             model_name=SYNTHETIC_DATA_GEN_MODEL,
                             prompt_for_synthetic_data_generation=PROMPT_FOR_SYNTHETIC_DATA_GENERATION,
                             pydantic_schema=Recipe,
                             output_file=SYNTHETIC_DATASET_FILE)
    create_embeddings()
    await ensure_database()
    await ensure_extensions()
    await create_tables(perform_drop=True)
    await create_indexes()
    await populate_db()

if __name__ == '__main__':
    asyncio.run(setup())