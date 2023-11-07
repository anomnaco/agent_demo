import sys
sys.path.append("utils")
from local_creds import *
from langchain.prompts import PromptTemplate
import json
import requests
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.utilities import SerpAPIWrapper
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.load.dump import dumps

request_url = f"https://{ASTRA_DB_ID}-{ASTRA_DB_REGION}.apps.astra.datastax.com/api/json/v1/{KEYSPACE}/{COLLECTION_NAME}"
request_headers = { 'x-cassandra-token': ASTRA_DB_APPLICATION_TOKEN,  'Content-Type': 'application/json'}

#langchain openai interface
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_similar_docs(query):
    embedding = list(embedding_model.embed_query(query))
    payload = json.dumps({"find": {"sort": {"$vector": embedding},"options": {"limit": 3}}})
    relevant_docs = requests.request("POST", request_url, headers=request_headers, data=payload).json()['data']['documents']
    docs_content = ["From " + str(row['document_id']) + ": " + str(row['answer']) for row in relevant_docs]
    content_block = "\n".join(docs_content)
    return content_block

@tool
def get_context_vector_db(query: str) -> str:
    """A vector store. Useful for answering queries about Datastax Astra and related topics. Input should be a question. Always used as the first tool."""
    return "Query: " + query + "\n Context: " + get_similar_docs(query)


tools = [get_context_vector_db] + load_tools(["serpapi"])
tools[0].name = "Document_Search"
#tools[1].name = "Assess_Context"
tools[1].name = "Web_Search"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are very powerful assistant, that always gives context for question answers. Before giving an answer to a question you first perform a document search to aquire context. Always start with a document search. Never start with web search. After using document search, you always assess whether the context provided helps to answer the question. If document search does not provide an answer, then you get context from a search engine. Do not assess context after searching the web. If you have found the answer, give the answer to the user. Give a detailed answer."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

agent_executor = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True, agent_instructions= "Use the 'Document_Search' tool first. Only use other tools after that.")

def invoke_agent(query):
    a = agent_executor.ainvoke({"input": query})
    result = {}
    last_tool = a["intermediate_steps"][-1][0].to_json()["kwargs"]["tool"]
    if last_tool == "_Exception":
        last_tool = a["intermediate_steps"][-2][0].to_json()["kwargs"]["tool"]

    result["last_tool"] = last_tool
    result["output"] = a["output"]
    return result
