# Define constants
SYNTHETIC_DATASET_FILE = "./recipes.csv"
EMBEDDING_DATASET_FILE = "./recipes_with_embeddings.parquet"
SYNTHETIC_DATA_GEN_MODEL = "gemini-2.5-pro"
GEMMA_EMBEDDING_MODEL = "google/embeddinggemma/transformers/embeddinggemma-300m"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_DIM=768
PROMPT_FOR_SYNTHETIC_DATA_GENERATION = """
  Generate a list of unique recipes from around the world:

  Requirements:
  - Exactly 200 UNIQUE recipes.
  - Each recipe must include: recipe_name, description, ingredients (array of strings), time_minutes (int), country (string), preparation (string).
  - Use a variety of countries/regions. Avoid duplicates by recipe_name.
  """