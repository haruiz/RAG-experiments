import asyncio
import logging
from typing import List, Union

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step, Event
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate

# Load environment variables
load_dotenv(verbose=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt templates for citation-based QA
CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. \n"
    "Example:\n"
    "Source 1:\nThe sky is red in the evening and blue in the morning.\n"
    "Source 2:\nWater is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources:\n"
    "{context_str}\nQuery: {query_str}\nAnswer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please refine the existing answer based solely on the provided sources. "
    "Cite sources where necessary, following this format:\n"
    "Example:\n"
    "Existing answer: {existing_answer}\n"
    "{context_msg}\n"
    "Query: {query_str}\nRefined Answer: "
)

DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 20


class RetrieverEvent(Event):
    """Event triggered after document retrieval."""

    nodes: List[NodeWithScore]


class CreateCitationsEvent(Event):
    """Event triggered after creating citations."""

    nodes: List[NodeWithScore]


class CitationQueryEngineWorkflow(Workflow):
    """Workflow for processing queries with retrieval-augmented generation (RAG)."""

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Union[RetrieverEvent, None]:
        """Retrieve relevant nodes based on the query."""
        query = ev.get("query")
        if not query:
            logger.warning("No query provided.")
            return None

        logger.info(f"Querying database: {query}")

        await ctx.set("query", query)

        if ev.index is None:
            logger.error("Index is empty. Load documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)

        logger.info(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def create_citation_nodes(self, ev: RetrieverEvent) -> CreateCitationsEvent:
        """Create granular citation nodes from retrieved text chunks."""
        nodes = ev.nodes
        new_nodes: List[NodeWithScore] = []

        text_splitter = SentenceSplitter(
            chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
        )

        for node in nodes:
            text_chunks = text_splitter.split_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE)
            )

            for idx, text_chunk in enumerate(text_chunks, start=len(new_nodes) + 1):
                text = f"Source {idx}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )
                new_node.node.text = text
                new_nodes.append(new_node)

        logger.info(f"Created {len(new_nodes)} citation nodes.")
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: CreateCitationsEvent) -> StopEvent:
        """Generate an AI response based on retrieved citations."""
        llm = OpenAI(model="gpt-4o-mini")
        query = await ctx.get("query", default=None)

        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )

        response = await synthesizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)


async def run_workflow():
    """Initialize the index and run the query workflow."""
    logger.info("Loading documents...")
    documents = SimpleDirectoryReader("downloads").load_data()

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    )

    logger.info("Running citation query workflow...")
    workflow = CitationQueryEngineWorkflow()
    result = await workflow.run(query="Write a blog post about agents?", index=index)

    bibliography = "\n\n### References\n"
    for node in result.source_nodes:
        bibliography += f"{node.get_text()}\n"
    print(bibliography)

    return result


if __name__ == "__main__":
    result = asyncio.run(run_workflow())
    print(result)
