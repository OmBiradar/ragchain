import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")

# Create prompt template
llm_prompt_template = """
Your name is RAGChain Assistant. Now the user question is: "{question}" """

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Create chain
rag_chain = (
    llm_prompt
    | llm
    | StrOutputParser()
)

def process_query(question):
    """Process a simple query with the LLM chain"""
    try:
        response = rag_chain.invoke({"question": question})
        return response
    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")
