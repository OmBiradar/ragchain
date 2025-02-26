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
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")

# Create prompt
llm_prompt_template = """
My name is {name}. Use the following context to answer the question.

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
        print(f"Loading existing database for {pdf_name}...")
        return load_existing_db(pdf_persist_dir)
    
    print(f"Processing {pdf_name}...")
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

# Update the main code
pdf_directory = "./HarryPotterBooks"
base_persist_directory = "./chroma_db"

# Get all retrievers
retrievers = load_all_pdfs(pdf_directory, base_persist_directory)

# Update the chain construction to use multiple retrievers
def merge_retrievers(question):
    all_contexts = []
    for retriever in retrievers:
        docs = retriever.invoke(question)
        all_contexts.extend([doc.page_content for doc in docs])
    return "\n".join(all_contexts)

# Update the RAG chain
rag_chain = (
    {
        "context": merge_retrievers,
        "question": RunnablePassthrough(),
        "name": lambda _: "Om Biradar"
    }
    | llm_prompt
    | llm
    | StrOutputParser()
)

# Test the chain
question = "What is Wizard Chess?" # Just a simple question :)
response = rag_chain.invoke(question)
print(response)
question = "How did Malfoys skin look when he asked for Voldemort's wand?" # Absurdly detailed question :D
response = rag_chain.invoke(question)
print(response)
