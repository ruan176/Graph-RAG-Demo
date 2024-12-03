import argparse
import logging

from graph_setup import setup_graph
from model_setup import get_template, get_structured_prompt
from query import query_llama

from localLLM import LocalLlamaLLM, RunnableLocalLlama
from langchain.chains import GraphCypherQAChain

# Configure the logger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s')

def main(question: str, use_graph):
    if question is None:
        logging.info("No question provided...")

    if use_graph:
        # Instantiate graph instance
        graph = setup_graph()

        # Graph embeddings and LLM query preparation
        text_template = get_template()
        formatted_prompt = get_structured_prompt(text_template)

        llm = RunnableLocalLlama(LocalLlamaLLM())

        cypherChain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                verbose=True,
                cypher_prompt=formatted_prompt,
                allow_dangerous_requests=True
            )

        # Run the user query.
        response = cypherChain.run(question)
        logging.info("Returned Graph RAG response:")
        logging.info(response)

    else:
        logging.info("Not using Graph as source of knowledge...")
        response = query_llama(question)
        logging.info("Returned LLM response (No Graph RAG):")
        logging.info(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single input argument 'question'.")
    parser.add_argument("question", type=str, help="The question wanting to be answered with G RAG.")
    parser.add_argument('-g', '--use-graph', action='store_true', help="Flag to use the graph for enhancing the LLM response.")

    args = parser.parse_args()

    main(question=args.question, use_graph=args.use_graph)