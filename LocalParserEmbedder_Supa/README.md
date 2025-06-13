# RAG System with Supabase and Google Gemini

This project implements a Retrieval Augmented Generation (RAG) system using Supabase for vector storage and Google Gemini for language model inferencing.

## Components

1. **Document Processing & Embedding** (`localEmbedder_supa.py`)
   - Processes PDF documents
   - Chunks text into manageable pieces
   - Embeds chunks using BAAI/bge-large-en-v1.5
   - Stores embeddings in Supabase

2. **RAG Implementation** (`rag_gemini.py`)
   - Embeds user queries using the same embedding model
   - Performs vector search in Supabase
   - Retrieves relevant document chunks
   - Uses Google Gemini to generate responses based on context

3. **Command Line Interface** (`app.py`)
   - Simple CLI application for user interaction
   - Clean Q&A interface with source information

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Install Poppler (for PDF processing):
   ```
   # macOS
   brew install poppler
   ```

3. Process documents (if not done already):
   ```
   python localEmbedder_supa.py
   ```

4. Run the Q&A system:
   ```
   # Using the main CLI application
   python app.py
   
   # Alternative direct access to the RAG system
   python rag_gemini.py
   ```

## Supabase Setup

The system expects a Supabase database with a `documents` table configured for vector similarity search. Make sure you have the following stored procedure in your Supabase SQL:

```sql
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1024),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
```

## Environment Variables

The system uses the following environment variables:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_KEY: Your Supabase API key
- GEMINI_API_KEY: Your Google Gemini API key

These are currently hardcoded in the scripts but can be moved to a .env file for better security.

## Note

This is a basic RAG implementation. For production use, consider:
- Better error handling
- Proper authentication
- Query pre-processing and post-processing
- Better context handling for the LLM
