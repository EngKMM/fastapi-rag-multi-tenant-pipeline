from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import(
    OllamaEmbeddingFunction
)

app = FastAPI()

client = chromadb.PersistentClient(path="./chroma_db")

my_embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=my_embedding_function, # type : ignore
)

class DocumentSubmission(BaseModel):
    user_name: str
    content: str

@app.post("/documents")
def add_document(submission: DocumentSubmission):
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]
    
    collection.add(
        ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source":"profile","user_name":submission.user_name,"chunk_index":i}
            for i in range(len(chunks))
        ],
    )

    return {
        "message":f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }

@app.get("/ask")
def ask(question: str, user: str = None):
    # this is the 'retrieval' part of RAG where we search our chromadb for the 2 most relevant chunks

    query_args = {
        "query_texts":[question],
        "n_results":2,
    }
    
    if user:
        query_args["where"] = {"user_name":user}
    
    results = collection.query(**query_args)

    if not results["documents"][0]:
        context=""

    else:
    # we seperated the chunks by breakline before now it's time to join em
        context = "\n\n".join(results["documents"][0])

    augmented_prompt = f"""
Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

    # finally we generate a response by sending the augmented prompt to the local LLM
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Return the answer with the context so users can verify the source
    return {
        "question":question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }
        
