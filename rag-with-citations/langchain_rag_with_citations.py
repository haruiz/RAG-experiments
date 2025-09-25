import itertools
import typing
from pathlib import Path
import arxiv
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from rich import print

# Load environment variables from .env file
load_dotenv()

# Constants
FAISS_INDEX_PATH = Path("faiss_index")        # Path to save/load the FAISS vector index
DOWNLOAD_FOLDER = Path("downloads")           # Directory to store downloaded PDFs


def fetch_arxiv_papers(query: str, max_results: int = 5) -> list[arxiv.Result]:
    """
    Searches and fetches papers from arXiv using a query string.

    Args:
        query (str): The search query for arXiv.
        max_results (int): Maximum number of results to fetch.

    Returns:
        list[arxiv.Result]: A list of arXiv results.
    """
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return list(arxiv.Client().results(search))


def process_papers(results: list[arxiv.Result], download_folder: Path) -> list:
    """
    Downloads PDFs from arXiv results, splits them semantically into chunks,
    and enriches them with metadata.

    Args:
        results (list[arxiv.Result]): The list of arXiv search results.
        download_folder (Path): Path to download the PDF files to.

    Returns:
        list: A list of chunked documents with metadata.
    """
    openai_embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(embeddings=openai_embeddings)
    download_folder.mkdir(exist_ok=True, parents=True)
    documents = []

    for index, result in enumerate(results):
        pdf_path = result.download_pdf(dirpath=str(download_folder))
        pdf_docs = PyPDFLoader(pdf_path).load_and_split(text_splitter)

        for doc in pdf_docs:
            doc.metadata.update({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "entry_id": result.entry_id.split('/')[-1],
                "year": result.published.year,
                "url": result.entry_id,
                "ref": f"[{index + 1}]"
            })
        documents.extend(pdf_docs)

    return documents


def load_or_create_faiss_index(query: str, index_path: Path, download_folder: Path) -> FAISS:
    """
    Loads an existing FAISS index or creates a new one from arXiv papers.

    Args:
        query (str): Query used to fetch arXiv papers.
        index_path (Path): Path to load or save the FAISS index.
        download_folder (Path): Folder to download papers into.

    Returns:
        FAISS: The loaded or newly created FAISS index.
    """
    openai_embeddings = OpenAIEmbeddings()

    if index_path.exists():
        print("Loading existing FAISS index...")
        return FAISS.load_local(str(index_path), openai_embeddings, allow_dangerous_deserialization=True)

    print("Creating a new FAISS index...")
    documents = process_papers(fetch_arxiv_papers(query), download_folder)
    faiss_index = FAISS.from_documents(documents, openai_embeddings)
    faiss_index.save_local(str(index_path))
    return faiss_index


def search_similar_documents(faiss_index: FAISS, query: str, top_k: int = 20) -> list:
    """
    Searches for the most similar documents in the FAISS index using vector similarity.

    Args:
        faiss_index (FAISS): The vector store index.
        query (str): The user's question/query.
        top_k (int): Number of top similar documents to return.

    Returns:
        list: A list of similar documents.
    """
    return faiss_index.similarity_search(query, k=top_k)


def generate_response(context_docs: list, question: str) -> str:
    """
    Generates an LLM-based response using the retrieved documents as context.

    Args:
        context_docs (list): List of documents retrieved from vector search.
        question (str): The user's question to answer.

    Returns:
        str: A generated response with in-text citations.
    """
    # Sort and format documents for contextual prompt input
    sorted_docs = sorted(context_docs, key=lambda doc: doc.metadata.get("ref", "Unknown"))
    formatted_context = [
        f"{doc.metadata['ref']} {doc.metadata['title']}: {doc.page_content}" for doc in sorted_docs
    ]

    prompt_template = PromptTemplate(
        template="""
        Write a blog post based on the user query.
        When referencing information from the context, cite the appropriate source(s) using their corresponding numbers.
        Each source has been provided with a number and a title.
        Every answer should include at least one source citation.
        If none of the sources are helpful, indicate that.

        ------
        {context}
        ------
        Query: {query}
        Answer:
        """,
        input_variables=["query", "context"]
    )

    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    chain = prompt_template | model | parser

    return chain.invoke({"context": formatted_context, "query": question})


def main():
    """
    Main function to:
    1. Fetch and/or load arXiv papers.
    2. Build or load a FAISS index.
    3. Search for relevant documents.
    4. Generate an answer.
    5. Print the result with bibliography.
    """
    query = "hallucination"
    question = "How to mitigate hallucination ?"

    # Step 1-2: Load/create index
    faiss_index = load_or_create_faiss_index(query, FAISS_INDEX_PATH, DOWNLOAD_FOLDER)

    # Step 3: Search relevant documents
    relevant_docs = search_similar_documents(faiss_index, question)

    # Step 4: Generate response
    response = generate_response(relevant_docs, question)
    print("\nGenerated Response:\n", response)

    # Step 5: Build and print references
    bibliography = "\n\n### References\n"
    sorted_docs = sorted(relevant_docs, key=lambda doc: doc.metadata.get("ref", "Unknown"))

    for doc_key, documents in itertools.groupby(sorted_docs, key=lambda doc: doc.metadata.get("ref", "Unknown")):
        doc = next(documents)
        bibliography += (
            f"{doc.metadata.get('ref', 'Unknown')} {doc.metadata.get('title', 'Unknown')}, "
            f"{', '.join(doc.metadata.get('authors', 'Unknown'))}, arXiv {doc.metadata.get('entry_id', 'Unknown')}, "
            f"{doc.metadata.get('year', 'Unknown')}. {doc.metadata.get('url', 'Unknown')}\n"
        )
    print(bibliography)


if __name__ == "__main__":
    main()
