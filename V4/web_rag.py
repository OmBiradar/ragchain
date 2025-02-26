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
from urllib.parse import urlparse

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
        """Perform Google search and load webpage contents"""
        search_results = self.search.results(query, num_results)
        urls = [result["link"] for result in search_results]
        for url in urls:
            domain = urlparse(url).netloc
            print(domain)
        
        # # Define your allowed domains
        # allowed_domains = {
        #     "wikipedia.org",
        #     "docs.python.org",
        #     "github.com",
        #     "stackoverflow.com",
        #     "medium.com",
        #     "arxiv.org",
        #     "bbc.com",
        #     "reuters.com",
        #     "theguardian.com",
        #     "nytimes.com",
        # }

        # # Filter the urls list to only include ones with allowed domains
        # filtered_urls = [
        #     url for url in urls
        #     if any(domain in urlparse(url).netloc for domain in allowed_domains)
        # ]

        # # Optionally print the filtered domains
        # for url in filtered_urls:
        #     print("Filtered domain:", urlparse(url).netloc)

        # # Replace the original list with the filtered one
        # urls = filtered_urls

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
                persist_directory="./web_chroma_db"
            )
        else:
            self.vectorstore.add_documents(splits)
            
        return splits

    def query(self, question: str) -> str:
        """Process query through RAG pipeline"""
        # First search and load relevant content
        self.search_and_load(question)
        
        # Create retriever function
        def retrieve_context(question):
            docs = self.vectorstore.similarity_search(question, k=3)
            return "\n".join([doc.page_content for doc in docs])

        # Create RAG chain
        rag_chain = (
            {
                "context": retrieve_context,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

# Example usage
if __name__ == "__main__":
    
    rag = WebRAG()
    
    # Test questions
    questions = [
        "How to switch to GNOME on Wayland in GDM in arch linux?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {rag.query(question)}")
