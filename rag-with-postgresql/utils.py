import time
from os import PathLike
from typing import Optional, Union, Sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import kagglehub
import numpy as np
import pandas as pd
import torch
from google import genai
from google.genai import types
import logging
import umap.umap_ as umap
from google.genai.types import InlinedEmbedContentResponse, SingleEmbedContentResponse
from pydantic import BaseModel, ConfigDict, field_validator
from sentence_transformers import SentenceTransformer
from sqlalchemy import Row
import plotly.express as px

logger = logging.getLogger(__name__)

class Recipe(BaseModel):
    recipe_name: str
    description: str
    ingredients: list[str]
    preparation: str
    time_minutes: int
    country: str

    model_config = ConfigDict(from_attributes=True)

    @field_validator("ingredients", mode="before")
    @classmethod
    def split_ingredients(cls, v):
        if isinstance(v, str):
            # split by comma and strip spaces
            return [x.strip() for x in v.split(",") if x.strip()]
        return v or []


class RecipeList(BaseModel):
    recipes: list[Recipe]

    class Config:
        from_attributes = True


def create_synthetic_dataset(model_name: str,
                             prompt_for_synthetic_data_generation: str,
                             pydantic_schema: type[BaseModel],
                             client: Optional[genai.Client] = None,
                             output_file: Optional[Union[PathLike, str]] = None
                             ) -> None:
    """Uses Gemini to create a synthetic dataset based on a provided prompt and schema.

    Args:
        model_name: The name of the Gemini model to use for the synthetic data generation.
        prompt_for_synthetic_data_generation: The prompt to guide the model to generate the dataset.
        pydantic_schema: The Pydantic model schema that defines the structure of each data entry.
        client: An optional pre-configured `genai.Client` instance. If not provided,
                a new client will be created with default settings.
        output_file: Optional path to save the generated dataset CSV.
    """
    if client is None:
        client = genai.Client()

    response = client.models.generate_content(
        model=model_name,
        contents=types.Part.from_text(text=prompt_for_synthetic_data_generation),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[pydantic_schema],
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )
    )
    # Parse the response into structured data using the provided Pydantic schema.
    structured_output = response.parsed
    if output_file:
        df = pd.DataFrame([item.model_dump() for item in structured_output])
        df.to_csv(output_file, index=False)
    return structured_output


def get_gemini_embeddings(
        texts: list[str],
        model_variant: str,
        output_dim: int = 768,
        client: Optional[genai.Client] = None) -> list[list[float]]:
    """Computes embeddings for a list of texts using the Gemini Embedding model in batch mode.

    This function sends a batch embedding request to the Gemini API. It's suitable for
    processing a large number of documents at once, which is more efficient than sending
    one request per document. The function will poll the API until the batch job is
    completed and then return the resulting embeddings.

    Args:
        texts: A list of strings to embed.
        model_variant: The specific Gemini embedding model variant to use (e.g., "gemini-1.5-embedding").
        output_dim: The desired dimensionality of the output embeddings. Default is 1536.
        client: An optional pre-configured `genai.Client` instance. If not provided,
                a new client will be created with default settings.

    Returns:
        A list of embeddings, where each embedding is a list of floats.

    Raises:
        RuntimeError: If the batch embedding job fails.
    """
    if client is None:
        client = genai.Client()
    # Create a batch embedding job request. This sends all texts to the API at once.
    batch_job = client.batches.create_embeddings(
        # Specify the model to use for generating embeddings.
        model=model_variant,
        # Define the source of the texts to be embedded. Here, we use an in-memory list.
        src=types.EmbeddingsBatchJobSource(
            # `inlined_requests` is used for providing the data directly in the request.
            inlined_requests=types.EmbedContentBatch(
                # Convert each text string into a `Part` object for the API.
                contents=[types.Part.from_text(text=text) for text in texts],
                # Configure the embedding task.
                config=types.EmbedContentConfig(
                    # `RETRIEVAL_DOCUMENT` is optimized for texts that will be stored and retrieved.
                    task_type="RETRIEVAL_DOCUMENT",
                    # Specify the desired dimension for the output embedding vectors.
                    output_dimensionality=output_dim
                )
            )
        ),
        # Configure the batch job itself, giving it a display name for identification.
        config=types.CreateEmbeddingsBatchJobConfig(
            display_name="embedding-batch-job"
        )
    )
    # Log the name of the created batch job for tracking.
    logging.info(f"Created batch job: {batch_job.name}")
    # Define the set of states that indicate the job has finished.
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    # Get the unique name of the job to use for polling.
    job_name = batch_job.name
    logger.info(f"Polling status for job: {job_name}")
    # Fetch the initial status of the batch job.
    batch_job = client.batches.get(name=job_name)  # Initial get
    # Loop and poll the job status until it reaches a completed state.
    while batch_job.state.name not in completed_states:
        logger.info(f"Current state: {batch_job.state.name}")
        # Wait for 30 seconds before checking the status again to avoid excessive polling.
        time.sleep(30)  # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    logger.info(f"Job finished with state: {batch_job.state.name}")
    # If the job failed, raise an error with the details.
    if batch_job.state.name == 'JOB_STATE_FAILED':
        raise RuntimeError(f"Batch job failed: {batch_job.error}")

    # Initialize an empty list to store the final embeddings.
    embeddings = []
    # Check if the job destination contains the inlined responses.
    if batch_job.dest and batch_job.dest.inlined_embed_content_responses:
        # Iterate through each response in the completed batch job.
        for content_response in batch_job.dest.inlined_embed_content_responses:
            content_response: InlinedEmbedContentResponse
            # If a specific text failed to embed, log the error and append an empty list.
            if content_response.error:
                logging.error(f"Error in content response: {content_response.error}")
                embeddings.append([])
                continue
            # Extract the successful embedding response.
            embed_response: SingleEmbedContentResponse = content_response.response
            # Append the embedding values (a list of floats) to our results.
            embeddings.append(embed_response.embedding.values)
    return embeddings


def get_gemma_embeddings(texts: list[str],
                         model_variant: str) -> list[list[float]]:
    """Computes embeddings for a list of texts using a local Gemma model.

    This function uses the `sentence-transformers` library to load a pre-trained
    Gemma embedding model from Kaggle Hub. It automatically uses a CUDA-enabled GPU
    if available, otherwise, it falls back to the CPU. It then encodes the provided
    list of texts into embedding vectors.

    Args:
        texts: A list of strings to embed.
        model_variant: The specific Gemma model variant to use (e.g., "gemma-3-small").

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    # Determine the computation device: 'cuda' for GPU if available, otherwise 'cpu'.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Download the Gemma model from Kaggle Hub and get the local path.
    model_id = kagglehub.model_download(model_variant)
    # Load the pre-trained SentenceTransformer model and move it to the selected device.
    model = SentenceTransformer(model_id).to(device=device)
    # Encode the list of texts into embedding vectors.
    candidate_embeddings = model.encode(texts)
    # Convert the resulting numpy array of embeddings to a standard Python list of lists.
    return candidate_embeddings.tolist()


def _as_str(x) -> str:
    """Robustly stringifies an object, handling lists, tuples, sets, and None."""
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        return ", ".join(map(str, x))
    return str(x)


def compute_projection(
    X: np.ndarray,
    method: str = "tsne",
    random_state: int = 42,
    perplexity: Optional[int] = None,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> np.ndarray:
    """
    Compute a 2D projection of embeddings.

    Args:
        X: Embedding matrix [n_samples, n_features]
        method: 'pca', 'tsne', or 'umap'
        random_state: random seed
        perplexity: only used in t-SNE
        umap_neighbors: UMAP neighbors
        umap_min_dist: UMAP min distance

    Returns:
        projections: 2D numpy array [n_samples, 2]
        subtitle: description of method used
    """
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        projections = reducer.fit_transform(X)
    elif method == "tsne":
        n = len(X)
        auto_perp = max(5, min(30, (n - 1) // 3))
        perp = perplexity if perplexity is not None else auto_perp
        perp = min(perp, max(5, n // 3 if n >= 6 else 5))
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
            n_iter_without_progress=300,
            metric="cosine"
        )
        projections = reducer.fit_transform(X)
    elif method == "umap":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=random_state,
        )
        projections = reducer.fit_transform(X)
    else:
        raise ValueError("method must be 'pca', 'tsne', or 'umap'")

    return projections


def make_plot_figure(rows: Sequence[Row], method: str = "tsne", *args, **kwargs) -> None:
    """
    Create and display a 2D plot of recipe embeddings using the specified dimensionality reduction method.
    Args:
        rows: List of rows from the database, each containing (embedding vector, country, recipe_name)
        method: 'pca', 'tsne', or 'umap'
        *args,
        **kwargs:
    Returns:
        None
    """
    # Convert to arrays/lists
    X = np.array([np.array(r[0], dtype=np.float32) for r in rows])
    names = [r[2] for r in rows]
    countries = [r[1] for r in rows]

    projections = compute_projection(
        X,
        method=method,
        *args,
        **kwargs
    )

    plot_data = {
        "x": projections[:, 0],
        "y": projections[:, 1],
        "country": names,
        "name": countries,
        "label": [f"<b>{nm}</b><br>Country: {cty}" for nm, cty in zip(names, countries)],
    }
    plot_data = pd.DataFrame.from_dict(plot_data)
    fig = px.scatter(
        plot_data,
        x="x",
        y="y",
        color="country",
        hover_name="name",
        text="name",  # 👈 this keeps the recipe name always visible
        labels={"x": "Component 1", "y": "Component 2"},
        title=f"Recipe Embeddings",
        opacity=0.85
    )

    # Adjust text appearance
    fig.update_traces(
        textposition="top center",  # 'top center', 'bottom right', etc.
        marker=dict(size=8, line=dict(width=0.5)),
        hovertemplate="%{customdata[0]}<extra></extra>",
        customdata=np.array([[lbl] for lbl in plot_data["label"]]),
    )

    # Subtle layout polish
    fig.update_layout(
        legend_title_text="Country",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
    )

    # Optional: save sharable HTML
    #fig.write_html(save_html_path, include_plotlyjs="cdn")
    fig.show()




