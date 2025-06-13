#!/usr/bin/env python3
"""
Document Q&A with RAG - Command Line Interface
This application uses Retrieval Augmented Generation (RAG) to answer questions about documents.
The system searches through embedded documents in Supabase and uses Google's Gemini to generate responses.
"""

from rag_gemini import run_rag
import sys


def main():
    """Main entry point for the command line RAG application"""
    print("=" * 80)
    print("📚 Document Q&A with RAG")
    print("=" * 80)
    print("This application uses Retrieval Augmented Generation (RAG) to answer questions about your documents.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("=" * 80)

    while True:
        try:
            # Get user query
            query = input("\nEnter your question: ")
            
            # Check for exit command
            if query.lower() in ['exit', 'quit']:
                print("\nThank you for using the Document Q&A system. Goodbye!")
                break
                
            if not query.strip():
                print("Please enter a valid question.")
                continue
                
            print("\nProcessing your query...\n")
            
            try:
                # Call the RAG system
                result = run_rag(query)
                
                # Display the results in a formatted way
                print("=" * 80)
                print("RESPONSE:")
                print("-" * 80)
                print(result["response"])
                print("\n" + "-" * 80)
                
                # Display information about sources
                print("SOURCES USED:")
                if result.get("sources"):
                    for i, source in enumerate(result["sources"], 1):
                        print(f"{i}. {source['source']} (Page: {source['page']}, Relevance: {source['relevance']})")
                else:
                    print("No sources retrieved")
                
                print(f"Total documents found: {result.get('results_count', 0)}")
                print("=" * 80)
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
        except KeyboardInterrupt:
            print("\nOperation canceled by user. Exiting...")
            break
            
    print("\nThank you for using the Document Q&A with RAG CLI.")
    print("Powered by Supabase Vector Database and Google Gemini")


if __name__ == "__main__":
    main()
