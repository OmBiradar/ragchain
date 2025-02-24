import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser

os.environ["GOOGLE_API_KEY"] = "AIzaSyBb_jB_IYGTcP_FeHSb8tPNSE-nkZ2I9yo"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")

llm_prompt_template = """
Your name is {name}. Now the user question is: "{question}" """

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Create chain
rag_chain = (
    llm_prompt
    | llm
    | StrOutputParser()
)

# Generate text
question = "Who is the president of the United States?"
response = rag_chain.invoke({"name": "Om Biradar", "question": question})
print(response)
