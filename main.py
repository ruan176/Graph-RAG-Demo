import argparse
import logging

from graph_setup import setup_graph
from model_setup import get_template, get_structured_prompt
from local_llm import LocalLlamaLLM, RunnableLocalLlama

from langchain.chains import GraphCypherQAChain

# Configure the logger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s')

def main(question: str, use_graph, verbose):
    """
    Main method to run Graph RAG LLM Demonstrator.
    
    params: 
        question: User question.
        use_graph: Tag to use graph RAG reference data source to 
                   generate response (Default False, graph not used.)
        verbose: Tag to print graph RAG intermediate steps when running
                 demonstrator (Defualt False, intermediate steps not shown)           
    """

    if question is None:
        logging.warning("No question provided...")

    if use_graph:
        # Instantiate graph connection instance
        graph = setup_graph()

        # Graph embeddings and LLM query preparation
        text_template = get_template()
        formatted_prompt = get_structured_prompt(text_template)

        # Define LLM connection
        llm = RunnableLocalLlama(LocalLlamaLLM())

        cypherChain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                verbose=verbose,
                cypher_prompt=formatted_prompt,
                allow_dangerous_requests=True
            )

        # Run the user query.
        response = cypherChain.run(question)
        print(response)
        logging.info(f"LLM Graph RAG Response: {response}")

    else:
        llm = LocalLlamaLLM()   
        response = llm._call(question)
        print(response)
        logging.info(f"LLM Response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single input argument 'question'.")
    parser.add_argument("question", type=str, help="The question wanting to be answered with G RAG.")
    parser.add_argument('-g', '--use-graph', action='store_true', help="Flag to use the graph for enhancing the LLM response.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to print intermediate RAG processing steps.")

    args = parser.parse_args()

    main(question=args.question, use_graph=args.use_graph, verbose=args.verbose)