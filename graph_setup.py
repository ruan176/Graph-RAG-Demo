from langchain_community.graphs import Neo4jGraph

def setup_graph() -> Neo4jGraph:
    """
    Setup and return a connection instance of Neo4jGraph 
    class.

    Amend variables below as per setup if running locally.
    """
    NEO4J_URI = 'bolt://localhost:7687'
    NEO4J_USERNAME = 'user1'
    NEO4J_PASSWORD = 'password'
    NEO4J_DATABASE = 'neo4j'

    return Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD, 
        database=NEO4J_DATABASE
    )



