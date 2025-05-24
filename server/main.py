import os
import sys
import logging
import time
from fastapi import FastAPI, HTTPException, Body, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import uvicorn
from server.utils import check_environment, get_env_var

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG models from different versions
from server.models import (
    semantic_analyzer,
    simple_llm,
    single_pdf_rag,
    multi_pdf_rag,
    web_search_rag,
    scholar_rag
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check environment variables
if not check_environment():
    logger.warning("Some required environment variables are missing.")

# Initialize FastAPI app
app = FastAPI(
    title="RAGChain API",
    description="""
    API for various RAG (Retrieval Augmented Generation) models.
    
    This API provides access to multiple types of RAG implementations:
    - Simple LLM query processing
    - Single PDF document RAG
    - Multiple PDF documents RAG
    - Web search augmented RAG
    - Google Scholar search
    - Semantic response analysis
    """,
    version="1.0.0",
    docs_url=None,
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_env_var("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
        self.window_seconds = 60  # 1 minute window
    
    def is_allowed(self):
        current_time = time.time()
        # Remove timestamps older than the window
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < self.window_seconds
        ]
        # Check if we're under the limit
        allowed = len(self.request_timestamps) < self.requests_per_minute
        if allowed:
            self.request_timestamps.append(current_time)
        return allowed

# Create rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=int(get_env_var("RATE_LIMIT", "60"))
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not rate_limiter.is_allowed():
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    response = await call_next(request)
    return response

# Custom OpenAPI docs
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="RAGChain API Documentation",
        swagger_favicon_url=""
    )

# API models with improved validation and documentation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The user's query text")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is retrieval augmented generation?"
            }
        }
    
class AnalyzeRequest(BaseModel):
    response: str = Field(..., min_length=1, description="The text response to analyze")
    expected: str = Field(..., min_length=1, description="The expected text to compare with")

    class Config:
        schema_extra = {
            "example": {
                "response": "The sky is blue",
                "expected": "The sky appears blue due to Rayleigh scattering"
            }
        }

class PDFRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The user's query text")
    pdf_name: str = Field(..., description="The name of the PDF file without extension")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the main theme of the book?",
                "pdf_name": "The-Alchemist"
            }
        }

class MultiPDFRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The user's query text")

    class Config:
        schema_extra = {
            "example": {
                "query": "Who is Harry Potter?"
            }
        }

class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The user's query text")
    num_results: int = Field(3, ge=1, le=10, description="Number of search results to retrieve")

    class Config:
        schema_extra = {
            "example": {
                "query": "Latest advances in RAG technology",
                "num_results": 3
            }
        }

class ScholarRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The academic topic to search for")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of scholarly articles to retrieve")

    class Config:
        schema_extra = {
            "example": {
                "query": "Large language models retrieval augmented generation",
                "max_results": 5
            }
        }

class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: List[str]

# API routes
@app.get("/", summary="API Root", description="Welcome message and API status")
async def root():
    return {"message": "Welcome to RAGChain API", "status": "active"}

@app.get("/health", response_model=HealthResponse, summary="Health Check", 
         description="Check the health status of the API and available models")
async def health_check():
    """
    Health check endpoint to verify the API and its components are functioning properly.
    Returns status information and a list of available models.
    """
    models_available = []
    
    try:
        # Test simple LLM availability
        simple_llm.process_query("test")
        models_available.append("simple_llm")
    except:
        pass
        
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_available": models_available
    }

@app.post("/v0/analyze", 
         summary="Analyze Response Similarity",
         description="Compare two text responses for semantic similarity using embeddings")
async def analyze_response(request: AnalyzeRequest):
    """
    Compare two text responses for semantic similarity.
    
    - **response**: The text response to analyze
    - **expected**: The expected text to compare with
    
    Returns "Yes" if responses are semantically similar, "No" if not.
    """
    try:
        result = semantic_analyzer.analyze_response(request.response, request.expected)
        return {
            "result": result,
            "response": request.response,
            "expected": request.expected
        }
    except Exception as e:
        logger.error(f"Error in semantic analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/v1/simple",
         summary="Simple LLM Query",
         description="Process a query using a simple LLM without any retrieval augmentation")
async def simple_query(request: QueryRequest):
    """
    Process a query with a simple LLM chain.
    
    This endpoint uses the LLM directly without any retrieval augmentation.
    
    - **query**: The user's text query
    
    Returns the generated response from the LLM.
    """
    start_time = time.time()
    try:
        response = simple_llm.process_query(request.query)
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "model": "gemini-2.0-pro-exp-02-05",
            "processing_time": round(processing_time, 2),
            "query": request.query
        }
    except Exception as e:
        logger.error(f"Error in simple LLM processing: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/v2/pdf",
         summary="Single PDF RAG Query",
         description="Query against a single PDF document using RAG")
async def pdf_query(request: PDFRequest):
    """
    Query against a single PDF document using RAG.
    
    This endpoint retrieves relevant information from a specified PDF document
    to answer the query using retrieval augmented generation.
    
    - **query**: The user's text query
    - **pdf_name**: The name of the PDF file without extension
    
    Returns the generated response based on PDF content.
    """
    start_time = time.time()
    try:
        response = single_pdf_rag.query_pdf(request.query, request.pdf_name)
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "pdf": request.pdf_name,
            "model": "gemini-2.0-pro-exp-02-05",
            "processing_time": round(processing_time, 2),
            "query": request.query
        }
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise HTTPException(status_code=404, detail=f"PDF file not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error in PDF RAG: {e}")
        raise HTTPException(status_code=500, detail=f"PDF query failed: {str(e)}")

@app.post("/v3/multi-pdf",
         summary="Multi PDF RAG Query",
         description="Query against multiple PDF documents using RAG")
async def multi_pdf_query(request: MultiPDFRequest):
    """
    Query against multiple PDF documents using RAG.
    
    This endpoint retrieves relevant information from multiple PDF documents
    to answer the query using retrieval augmented generation.
    
    - **query**: The user's text query
    
    Returns the generated response based on content from multiple PDFs.
    """
    start_time = time.time()
    try:
        response = multi_pdf_rag.query_multiple_pdfs(request.query)
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "model": "gemini-2.0-pro-exp-02-05",
            "processing_time": round(processing_time, 2),
            "query": request.query,
            "source": "Multiple PDFs (Harry Potter books)"
        }
    except FileNotFoundError as e:
        logger.error(f"PDF directory not found: {e}")
        raise HTTPException(status_code=404, detail=f"PDF directory not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error in Multi-PDF RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-PDF query failed: {str(e)}")

@app.post("/v4/web-search",
         summary="Web Search RAG Query",
         description="Query with web search augmentation using RAG")
async def web_search_query(request: WebSearchRequest):
    """
    Query with web search augmentation using RAG.
    
    This endpoint searches the web for relevant information and uses that 
    to answer the query with retrieval augmented generation.
    
    - **query**: The user's text query
    - **num_results**: Number of search results to retrieve (1-10)
    
    Returns the generated response based on web search results.
    """
    start_time = time.time()
    try:
        response = web_search_rag.query(request.query, request.num_results)
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "model": "gemini-2.0-pro-exp-02-05",
            "processing_time": round(processing_time, 2),
            "query": request.query,
            "num_results": request.num_results,
            "source": "DuckDuckGo Web Search"
        }
    except Exception as e:
        logger.error(f"Error in Web Search RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Web search query failed: {str(e)}")

@app.post("/v5/scholar",
         summary="Google Scholar Search",
         description="Search for academic articles on Google Scholar")
async def scholar_query(request: ScholarRequest):
    """
    Search for academic articles on Google Scholar.
    
    This endpoint retrieves academic articles related to the query from Google Scholar.
    
    - **query**: The academic topic to search for
    - **max_results**: Maximum number of scholarly articles to retrieve (1-20)
    
    Returns a list of scholarly articles with title, authors, abstract, and publication year.
    """
    start_time = time.time()
    try:
        articles = scholar_rag.fetch_articles(request.query, request.max_results)
        processing_time = time.time() - start_time
        
        return {
            "articles": articles,
            "processing_time": round(processing_time, 2),
            "query": request.query,
            "num_results": len(articles),
            "source": "Google Scholar"
        }
    except Exception as e:
        logger.error(f"Error in Scholar search: {e}")
        raise HTTPException(status_code=500, detail=f"Scholar query failed: {str(e)}")

# API startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting RAGChain API server...")
    # Check environment variables
    if not check_environment():
        logger.warning("Some required environment variables are missing!")
    logger.info("Server started successfully")

# API shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RAGChain API server...")

if __name__ == "__main__":
    # Get configuration from environment variables with defaults
    host = get_env_var("HOST", "0.0.0.0")
    port = int(get_env_var("PORT", "8000"))
    reload = get_env_var("RELOAD", "True").lower() == "true"
    
    # Log startup configuration
    logger.info(f"Starting server on {host}:{port} (reload={reload})")
    
    # Run server
    uvicorn.run(
        "server.main:app", 
        host=host, 
        port=port, 
        reload=reload,
        log_level="info"
    )
