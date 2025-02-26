import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class SemanticAnalyzer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.output_parser = StrOutputParser()
        
    def get_embedding(self, text):
        if not isinstance(text, str):
            text = str(text)
        return self.embeddings.embed_query(text)
    
    def compare_responses(self, response1, response2):
        emb1 = self.get_embedding(response1)
        emb2 = self.get_embedding(response2)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return similarity > 0.8  # Return True if responses are semantically similar
    
    def analyze_response(self, response, expected):
        return "Yes" if self.compare_responses(response, expected) else "No"

# Create analyzer
analyzer = SemanticAnalyzer()

# Test cases with expected responses
test_cases = [
    (
        "Why does the sky appear blue? Answer in 1 sentence.",
        "The sky appears blue because shorter blue wavelengths scatter more in the atmosphere than other colors."
    ),
    (
        "Are cats capable of flight under normal conditions? Answer in 1 sentence.",
        "No, cats are terrestrial animals and cannot fly naturally."
    ),
    (
        "Is it true that 2+2 equals 4? Answer in 1 sentence.",
        "Yes, in standard arithmetic, the sum of 2 and 2 is 4."
    )
]

# Generate and analyze responses
for question, expected in test_cases:
    # Parse LLM response to string
    response = str(analyzer.llm.invoke(question))
    is_similar = analyzer.analyze_response(response, expected)
    print(f"Q: {question}")
    print(f"Response: {response}")
    print(f"Similarity result: {is_similar}\n")
