from langchain.prompts.prompt import PromptTemplate


def get_template() -> str:
    """
    Get prompt template for LLM, used to generate the required Cypher query that 
    provides additional context to response to user question.
    """

    CYPHER_GENERATION_TEMPLATE = """
    Task: Create a Cypher query to interact with a graph database.
    Instructions:
    Use only the relationship types and properties specified in the given schema.
    Do not incorporate any other relationship types or properties not explicitly included in the schema.
    Schema:
    {schema}

    Note:
    Only output the Cypher query.
    Do not provide explanations, apologies, or respond to unrelated questions.
    Examples: Below are sample Cypher queries for specific questions:

    # How many actors were in the movie 'The Matrix'?
    MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie) 
    WHERE movie.title = 'The Matrix' 
    RETURN COUNT(actor)
    The question is:
    {question}
    """

    return CYPHER_GENERATION_TEMPLATE

def get_structured_prompt(template) -> PromptTemplate:
    """
    Structure prompt in format for langchain chain used to
    structure Graph RAG response.
    """
    CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=template)

    return CYPHER_GENERATION_PROMPT
