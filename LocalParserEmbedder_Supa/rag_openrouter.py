import os
import requests
import json
from langchain.embeddings import HuggingFaceBgeEmbeddings
from supabase import create_client, Client
import torch

# Load environment variables and API keys
SUPABASE_URL = "https://rrzxyhgddphzsnnsfhjx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyenh5aGdkZHBoenNubnNmaGp4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3MTM2ODgsImV4cCI6MjA2NTI4OTY4OH0.S-VOCbAzsOyOUlIzi6sjypOt2B-E57Lf3Huau6LJKc0"

# OpenRouter API configuration
OPENROUTER_API_KEY = "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"  # Replace with your actual OpenRouter API key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def generate_response(query, context, model_name="anthropic/claude-3.5-sonnet"):
    """Generate a response using OpenRouter API based on the query and context"""
    
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
        # OpenRouter API call
        api_url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Optional: for analytics
            "X-Title": "RAG System with Supabase"  # Optional: for analytics
        }
        
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            try:
                return response_json['choices'][0]['message']['content']
            except (KeyError, IndexError):
                return "Retrieved content successfully, but couldn't parse the response format. Please check the API response structure."
        else:
            return f"Error: API call failed with status code {response.status_code}. Response: {response.text[:200]}..."
            
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I couldn't generate a response due to an error: {str(e)}"

def list_available_models():
    """List available models from OpenRouter"""
    try:
        api_url = f"{OPENROUTER_BASE_URL}/models"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable OpenRouter Models:")
            print("-" * 50)
            for model in models.get('data', [])[:20]:  # Show first 20 models
                name = model.get('id', 'Unknown')
                description = model.get('name', 'No description')
                print(f"• {name}: {description}")
            print("-" * 50)
            print(f"Total models available: {len(models.get('data', []))}")
        else:
            print(f"Failed to fetch models: {response.status_code}")
            
    except Exception as e:
        print(f"Error fetching models: {e}")

def run_rag(query, model_name="anthropic/claude-3.5-sonnet"):
    """Run the complete RAG pipeline"""
    # 1. Embed the query
    query_embedding = embed_query(query)
    
    # 2. Perform vector search
    search_results = vector_search(query_embedding)
    
    # 3. Format the context from search results
    context = format_context(search_results)
    
    # 4. Generate response using LLM
    response = generate_response(query, context, model_name)
    
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
        "model_used": model_name,
        "results_count": len(search_results),
        "response": response,
        "sources": sources
    }

# Interactive CLI
if __name__ == "__main__":
    print("RAG System with Supabase Vector DB and OpenRouter API")
    print("=" * 60)
    
    # Check if API key is set
    if OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        print("⚠️  WARNING: Please set your OpenRouter API key in the OPENROUTER_API_KEY variable")
        print("You can get one from: https://openrouter.ai/")
        print()
    
    print("Available commands:")
    print("• Type your question to get an answer")
    print("• Type 'models' to see available models")
    print("• Type 'change_model' to switch models")
    print("• Type 'exit' to quit")
    print()
    
    # Default model
    current_model = "anthropic/claude-3.5-sonnet"
    print(f"Current model: {current_model}")
    print("-" * 60)
    
    while True:
        user_input = input("\nEnter your command or question: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "models":
            list_available_models()
        elif user_input.lower() == "change_model":
            print("\nPopular models you can try:")
            popular_models = [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o",
                "openai/gpt-4o-mini", 
                "google/gemini-pro-1.5",
                "meta-llama/llama-3.1-8b-instruct",
                "mistralai/mixtral-8x7b-instruct",
                "cohere/command-r-plus"
            ]
            for i, model in enumerate(popular_models, 1):
                print(f"{i}. {model}")
            
            new_model = input(f"\nEnter model name (or press Enter to keep '{current_model}'): ").strip()
            if new_model:
                current_model = new_model
                print(f"✅ Model changed to: {current_model}")
            else:
                print(f"✅ Keeping current model: {current_model}")
        elif user_input:
            print("\n🔍 Processing your question...")
            try:
                result = run_rag(user_input, current_model)
                print("\n" + "="*80)
                print("RESPONSE:")
                print("-"*80)
                print(result["response"])
                print("\n" + "-"*80)
                print(f"MODEL USED: {result['model_used']}")
                print(f"DOCUMENTS FOUND: {result['results_count']}")
                print("SOURCES USED:")
                if result["sources"]:
                    for i, source in enumerate(result["sources"], 1):
                        print(f"{i}. {source['source']} (Page: {source['page']}, Relevance: {source['relevance']})")
                else:
                    print("No sources retrieved")
                print("="*80)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("Please enter a valid command or question.")
