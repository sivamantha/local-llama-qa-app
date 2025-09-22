import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
# Import the Ollama LLM
from llama_index.llms.ollama import Ollama

# Import the embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure the LLM to use your local Ollama model
# The request_timeout is increased to give the local model more time to respond.
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
print("✅ LLM Configured to use local Llama 3!")

# Configure the embedding model
# Settings.llm = Ollama(model="llama3", base_url="http://10.255.255.254:11434", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("✅ Embedding Model Configured!")

# Load your data from the 'data' directory
try:
    documents = SimpleDirectoryReader("data").load_data()
    print(f"✅ Loaded {len(documents)} document(s).")
except Exception as e:
    print(f"Error loading documents: {e}")
    exit()

# Create an index from the loaded documents
# The index is a searchable data structure of your documents.
index = VectorStoreIndex.from_documents(documents)
print("✅ Index created successfully.")

# Create a query engine from the index
# This is the main tool for asking questions.
query_engine = index.as_query_engine()
print("🚀 Query engine ready. You can now ask questions!")


# Start the interactive Q&A loop
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")

    if query.lower() == 'exit':
        print("Goodbye!")
        break

    # Get the response from the query engine
    response = query_engine.query(query)

    # Print the response
    print("\nAnswer:")
    print(response)