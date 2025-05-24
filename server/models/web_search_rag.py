from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

class WebRAG:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.search = DuckDuckGoSearchAPIWrapper()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize vectorstore path
        self.db_path = "./web_chroma_db"
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize vectorstore 
        try:
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        except:
            # Create a new vectorstore if one doesn't exist
            self.vectorstore = None

        # Create prompt template
        self.prompt = PromptTemplate.from_template("""
        Use the following context from web searches to answer the question.
        If the information suggests more than 1 answer, provide all of them along with the context.
        If the answer cannot be found in the context, say "I couldn't find relevant information."

        Context: {context}

        Question: {question}
        """)

    def search_and_load(self, query: str, num_results: int = 3) -> List[str]:
        """Perform search and load webpage contents"""
        search_results = self.search.results(query, num_results)
        urls = [result["link"] for result in search_results]
        
        # Load webpages
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create or update vectorstore
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
        else:
            self.vectorstore.add_documents(splits)
            
        return splits

# Create singleton instance
web_rag = WebRAG()

def query(question: str, num_results: int = 3) -> str:
    """Process query through RAG pipeline"""
    # First search and load relevant content
    web_rag.search_and_load(question, num_results)
    
    # Create retriever function
    def retrieve_context(question):
        docs = web_rag.vectorstore.similarity_search(question, k=3)
        return "\n".join([doc.page_content for doc in docs])

    # Create RAG chain
    rag_chain = (
        {
            "context": retrieve_context,
            "question": RunnablePassthrough()
        }
        | web_rag.prompt
        | web_rag.llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
