from langchain.prompts.prompt import PromptTemplate


def get_template():
    """
    Get template used to prompt llm to return the required Cypher query to provide
    additional context to response to user question.
    """

    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to 
    query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the 
    schema. Do not use any other relationship types or properties that 
    are not provided.
    Schema:
    {schema}
    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than 
    for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    Examples: Here are a few examples of generated Cypher 
    statements for particular questions:

    # How many actors were in the movie 'The Matrix'?
    MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie) 
    WHERE movie.title = 'The Matrix' 
    RETURN COUNT(actor)
    The question is:
    {question}"""

    return CYPHER_GENERATION_TEMPLATE

def get_structured_prompt(template):
    """
    Structure prompt in format for langchain chain used to
    structure Graph RAG response.
    """
    CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=template)

    return CYPHER_GENERATION_PROMPT
