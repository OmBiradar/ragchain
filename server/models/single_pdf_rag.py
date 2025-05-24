import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")

# Create prompt
llm_prompt_template = """
My name is RAGChain Assistant. Use the following context to answer the question.

Context: {context}

Question: {question}"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Load and process PDF
def load_and_process_pdf(pdf_path, persist_directory):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

# To load existing database
def load_existing_db(persist_directory="./chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def query_pdf(question, pdf_name="The-Alchemist"):
    """Query against a single PDF document"""
    # Check if PDF exists in project
    pdf_path = f"../V2/{pdf_name}.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    persist_directory = f"./chroma_db/{pdf_name}"

    # Use existing database if available
    if os.path.exists(persist_directory):
        vectorstore = load_existing_db(persist_directory)
    else:
        os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
        vectorstore = load_and_process_pdf(pdf_path, persist_directory)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
     
    # Create the RAG chain
    rag_chain = (
        {
            "context": retriever, 
            "question": RunnablePassthrough()
        }
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    # Generate response
    response = rag_chain.invoke(question)
    return response
