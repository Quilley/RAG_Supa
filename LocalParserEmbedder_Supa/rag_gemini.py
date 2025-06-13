import os
import requests
import json
from langchain.embeddings import HuggingFaceBgeEmbeddings
from supabase import create_client, Client
import google.generativeai as genai
import torch

# Load environment variables and API keys
SUPABASE_URL = "https://rrzxyhgddphzsnnsfhjx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyenh5aGdkZHBoenNubnNmaGp4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3MTM2ODgsImV4cCI6MjA2NTI4OTY4OH0.S-VOCbAzsOyOUlIzi6sjypOt2B-E57Lf3Huau6LJKc0"
GEMINI_API_KEY = "AIzaSyD4Z4jF1eBcY-VqohFrDxKlqlo9EDP6sVg"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Google Gemini API
# Note: Configuration is handled directly in the API calls

# Set up the embedding model (same as used for document embedding)
# Detect Apple Silicon and set device accordingly
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def embed_query(query_text):
    """Embed the query using the same model as the documents"""
    return embedding_model.embed_query(query_text)

def vector_search(query_embedding, top_k=5):
    """
    Perform vector search in Supabase
    Returns top_k most similar documents
    """
    # Query the documents table with vector search
    response = supabase.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'match_threshold': 0.5,  # Adjust as needed
            'match_count': top_k
        }
    ).execute()
    
    # Return the results
    return response.data

def format_context(search_results):
    """Format the search results into context for the LLM"""
    if not search_results:
        return "No relevant information found in the knowledge base."
    
    # Sort results by similarity score to prioritize most relevant content
    sorted_results = sorted(search_results, key=lambda x: x.get('similarity', 0), reverse=True)
    
    context_parts = []
    
    for idx, result in enumerate(sorted_results):
        content = result.get('content', '')
        similarity = result.get('similarity', 0)
        source = result.get('metadata', {}).get('file_path', 'Unknown source')
        page = result.get('metadata', {}).get('page_number', 'N/A')
        
        # Format each result with its source and more detailed metadata
        header = f"DOCUMENT {idx+1} [Relevance: {similarity:.2f}]"
        source_info = f"Source: {source} | Page: {page}"
        separator = "-" * 40
        
        context_parts.append(f"{header}\n{source_info}\n{separator}\n{content}\n{separator}\n")
    
    # Join all formatted results
    final_context = "\n".join(context_parts)
    
    # Add a note about the number of sources
    final_context = f"The following information comes from {len(sorted_results)} document fragments:\n\n{final_context}"
    
    return final_context

def generate_response(query, context, model_name="gemini-1.5-pro"):
    """Generate a response using Google Gemini based on the query and context"""
    import requests
    import json
    
    # Create the prompt with query and retrieved context
    prompt = f"""
    You are a knowledgeable research assistant. Your task is to provide a comprehensive, 
    coherent, and well-structured answer to the user's question based ONLY on the provided context.
    
    Context information is below.
    ---------------------
    {context}
    ---------------------
    
    Given this information, answer the user's query: "{query}"
    
    Instructions for responding:
    1. Synthesize information from ALL relevant documents to create a complete answer.
    2. Connect ideas across different documents to provide a cohesive response.
    3. Structure your response with clear paragraphs that flow logically.
    4. Include specific details from the documents to support your answer.
    5. If the information is incomplete, explain what aspects are missing.
    6. If the query cannot be answered from the provided context, clearly state: "I don't have enough information to answer this question."
    7. DO NOT introduce external information not present in the context.
    8. DO NOT prefix your answer with phrases like "Based on the context" or similar.
    
    Answer:
    """
    
    try:
        # Direct API call approach that doesn't depend on SDK version
        api_url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            try:
                return response_json['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                # If we can't find the expected structure, return a more helpful message
                return "Retrieved content successfully, but couldn't parse the response format. Please check the API version compatibility."
        else:
            return f"Error: API call failed with status code {response.status_code}. Response: {response.text[:200]}..."
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I couldn't generate a response due to an error: {str(e)}"

def run_rag(query):
    """Run the complete RAG pipeline"""
    # 1. Embed the query
    query_embedding = embed_query(query)
    
    # 2. Perform vector search
    search_results = vector_search(query_embedding)
    
    # 3. Format the context from search results
    context = format_context(search_results)
    
    # 4. Generate response using LLM
    response = generate_response(query, context)
    
    # 5. Extract source information for transparency
    sources = []
    for result in search_results:
        source = result.get('metadata', {}).get('file_path', 'Unknown source')
        page = result.get('metadata', {}).get('page_number', 'N/A')
        similarity = result.get('similarity', 0)
        sources.append({
            "source": source,
            "page": page,
            "relevance": f"{similarity:.2f}"
        })
    
    return {
        "query": query,
        "results_count": len(search_results),
        "response": response,
        "sources": sources
    }

# Interactive CLI
if __name__ == "__main__":
    print("RAG System with Supabase Vector DB and Google Gemini")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        print("\nProcessing...")
        try:
            result = run_rag(query)
            print("\n" + "="*80)
            print("RESPONSE:")
            print("-"*80)
            print(result["response"])
            print("\n" + "-"*80)
            print("SOURCES USED:")
            if result["sources"]:
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['source']} (Page: {source['page']}, Relevance: {source['relevance']})")
            else:
                print("No sources retrieved")
            print("="*80)
        except Exception as e:
            print(f"Error: {e}")
