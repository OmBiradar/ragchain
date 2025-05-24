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

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")

# Create prompt
llm_prompt_template = """
My name is RAGChain Assistant. Use the following context to answer the question.

Context: {context}

Question: {question}"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# To load existing database
def load_existing_db(persist_directory="./chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def process_single_pdf(pdf_path, persist_directory):
    # Get PDF filename without extension for the folder name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_persist_dir = os.path.join(persist_directory, pdf_name)
    
    if os.path.exists(pdf_persist_dir):
        return load_existing_db(pdf_persist_dir)
    
    os.makedirs(pdf_persist_dir, exist_ok=True)
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=pdf_persist_dir
    )
    
    return vectorstore

def load_all_pdfs(pdf_directory, base_persist_directory):
    vectorstores = []
    
    # Create base persist directory if it doesn't exist
    os.makedirs(base_persist_directory, exist_ok=True)
    
    # Process each PDF individually
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            vectorstore = process_single_pdf(pdf_path, base_persist_directory)
            vectorstores.append(vectorstore)
    
    # Simplified retriever configuration with only supported parameters
    retrievers = [
        vs.as_retriever(
            search_kwargs={"k": 2}
        ) for vs in vectorstores
    ]
    
    return retrievers

# Function to merge retriever results
def merge_retrievers(retrievers, question):
    all_contexts = []
    for retriever in retrievers:
        docs = retriever.invoke(question)
        all_contexts.extend([doc.page_content for doc in docs])
    return "\n".join(all_contexts)

def query_multiple_pdfs(question):
    """Query against multiple PDF documents"""
    # Update paths according to server's file organization
    pdf_directory = "../V3/HarryPotterBooks"
    base_persist_directory = "./chroma_db/multi_pdf"
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    # Get all retrievers
    retrievers = load_all_pdfs(pdf_directory, base_persist_directory)
    
    if not retrievers:
        raise Exception("No PDF files found to process")
    
    # Create context function that merges all retriever results
    def get_context(q):
        return merge_retrievers(retrievers, q)
    
    # Update the RAG chain
    rag_chain = (
        {
            "context": get_context,
            "question": RunnablePassthrough()
        }
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    # Generate response
    response = rag_chain.invoke(question)
    return response
