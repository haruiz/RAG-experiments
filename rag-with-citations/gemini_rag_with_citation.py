import itertools
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    Tool,
    GoogleSearch,
    GroundingChunk
)

load_dotenv()

class Citation(BaseModel):
    """Represents a citation extracted from the Gemini grounding metadata."""
    title: str
    score: float
    link: str
    chunk_index: int
    chunk_text: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


def generate_content(prompt: str, model: str) -> genai.types.GenerateContentResponse:
    """
    Generates a text response using the specified Gemini model and a Google Search tool.

    Args:
        prompt (str): The input text or question.
        model (str): The model name to use.

    Returns:
        genai.types.GenerateContentResponse: The model's response including text and grounding data.
    """
    client = genai.Client()
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            response_modalities=["TEXT"],
            tools=[Tool(google_search=GoogleSearch())],
        ),
    )


def extract_citations(response: genai.types.GenerateContentResponse) -> list[Citation]:
    """
    Extracts citation information from a Gemini response with grounding metadata.

    Args:
        response (genai.types.GenerateContentResponse): The response from Gemini.

    Returns:
        list[Citation]: A list of extracted citation data.
    """
    citations = []
    grounding_metadata = response.candidates[0].grounding_metadata
    for support in grounding_metadata.grounding_supports:
        for idx, score in zip(support.grounding_chunk_indices, support.confidence_scores):
            chunk: GroundingChunk = grounding_metadata.grounding_chunks[idx]
            citations.append(
                Citation(
                    title=chunk.web.title,
                    link=chunk.web.uri,
                    score=score,
                    chunk_index=idx,
                    chunk_text=support.segment.text,
                    start_index=support.segment.start_index,
                    end_index=support.segment.end_index,
                )
            )
    return citations


def inject_citations_into_text(text: str, citations: list[Citation]) -> str:
    """
    Inserts inline citation references into the generated text.

    Args:
        text (str): The generated text.
        citations (list[Citation]): A list of citations with start/end positions.

    Returns:
        str: The text with inline citation markers (e.g., [1], [2,3]).
    """
    citations.sort(key=lambda x: (x.start_index, x.end_index))
    offset = 0
    for (start, end), group in itertools.groupby(citations, key=lambda x: (x.start_index, x.end_index)):
        group_list = list(group)
        indices = ",".join(str(c.chunk_index + 1) for c in group_list)
        citation_str = f"[{indices}]"
        text = text[:end + offset] + citation_str + text[end + offset:]
        offset += len(citation_str)
    return text


def format_citation_section(citations: list[Citation]) -> str:
    """
    Formats the citations into a markdown-style bibliography section.

    Args:
        citations (list[Citation]): A list of citation objects.

    Returns:
        str: A formatted string listing all citations.
    """
    result = "\n\n**Citations**\n\n"
    sorted_citations = sorted(citations, key=lambda x: x.chunk_index)
    for chunk_index, group in itertools.groupby(sorted_citations, key=lambda x: x.chunk_index):
        citation = list(group)[0]
        result += f"[{chunk_index + 1}] {citation.title} - {citation.link}\n"
    return result


def main():


    # Constants
    MODEL_NAME = "gemini-2.0-flash"
    response = generate_content("Write a blog post about Agents", MODEL_NAME)
    citations = extract_citations(response)

    generated_text = response.text
    final_text = inject_citations_into_text(generated_text, citations)
    final_text += format_citation_section(citations)
    print(final_text)


if __name__ == '__main__':
    main()
