import os
from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, Settings,)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Setup LLM (using local Ollama)
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0)

# Setup embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load document
try:
    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} docs from data/")
except Exception as e:
    print("Couldn't load docs:", e)
    exit()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create a query engine 
query_engine = index.as_query_engine()
print("Ask questions!")

# Start the interactive Q&A loop 
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")

    if query.lower() == 'exit':
        print("bye!")
        break

    response = query_engine.query(query)

    print("\nAnswer:")
    print(response)