import chromadb
from chromadb.utils.embedding_functions import ollama_embedding_function
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction
)

# we're gonna load the profile document we created
with open("profile.txt", "r") as profile_doc:
    text = profile_doc.read()

# here we're splitting our text into chunks and each chunk is seperated by break-line yum yum!
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


# next we're gonna initalize chromaDB with persistent client which saves data to disk
client = chromadb.PersistentClient(path="./chroma_db")

# connect to the ollama embedding model to embed our text (whole point of this)
my_embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text", # name of our embedding model
    url="http://localhost:11434", # the local port ollama runs on
)

# Create a collection
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=my_embedding_function # type : ignore
)

# let's add chunks to the collection
collection.add(
    ids=[f"chunks{i}" for i in range(len(chunks))], #ids
    documents=chunks, #text docs
    metadatas=[{"source": "profile", "chunk_index": i} for i in range(len(chunks))],
)

print("knowledge base built successfully")



