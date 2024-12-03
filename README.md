# Graph RAG (Graph-augmented Retrieval with LLM)

Graph RAG is an innovative application that enhances the capabilities of a Large Language Model (LLM) by querying a knowledge graph with relevant data to improve its response quality. Users can input their questions, and the system combines the power of the LLM and knowledge graph to provide richer, more accurate answers.

## Features

- **Query Knowledge Graph**: Automatically retrieves relevant data from a knowledge graph using few-shot training.
- **LLM-based Responses**: Leverages a large language model to process the question and integrate knowledge from the graph.
- **Dynamic Questioning**: Allows users to ask any question and receive enhanced answers.
- **Easy to Use**: Simply run the `main.py` file and pass your question through the command line argument.

## Requirements
  
  To install the necessary dependencies, run:

  ```bash
  pip install -r requirements.txt
  ```

## Further Work
preprocess natural text data into a graph format using chunking and embeddings to enable effective value extraction.

local host the llm to create a completely offline application, useful for scenarios with sensitive data.


