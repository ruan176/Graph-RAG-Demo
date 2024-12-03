from langchain_community.graphs import Neo4jGraph

def setup_graph():
    """
    Setup and return a local Neo4j instance for hosting the reference graph.

    Modify the variables below as required.
    """
    NEO4J_URI = 'bolt://localhost:7687'
    NEO4J_USERNAME = 'user'
    NEO4J_PASSWORD = 'password'
    NEO4J_DATABASE = 'neo4j'

    return Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD, 
        database=NEO4J_DATABASE
    )



