<img src="https://cdn.prod.website-files.com/677c400686e724409a5a7409/6790ad949cf622dc8dcd9fe4_nextwork-logo-leather.svg" alt="NextWork" width="300" />

# Build a RAG API with FastAPI

**Project Link:** [View Project](http://learn.nextwork.org/projects/ai-devops-api)

**Author:** Kareem Gohary  
**Email:** karimgohary.gohary@gmail.com

---

---

## Introducing Today's Project!

In this project, I'm going to be building a RAG API for my local LLM. This will help me better understand how enterprise level engineers apply RAG APIs to production models to increase performance.

### Key tools and concepts

The key tools I used include Ollama, FastAPI, ChromaDB, Nomic-embed-text. Key concepts I learnt include building a RAG pipeline, creating vector embeddings, and using metadata filtering to support multiple users.

### Challenges and wins

This project took me approximately 2-3 hours. The most challenging part was getting the multi-user filtering to work correctly. I initially had a bug where I was trying to apply the metadata filter after the database query had already executed. I had to figure out how to restructure the code to build the query arguments first, dynamically add the username filter, and then unpack those arguments into the search.

---

## Performing RAG Manually

I'm going to start with building my own manual RAG pipeline. I implemented RAG manually by including some personal info in the prompt itself just to test out how it works. 

![Image](http://learn.nextwork.org/confident_white_wise_huckleberry/uploads/ai-devops-api_v3j7x5b9)

### Understanding the three parts of RAG

RAG stands for Retrieval-Augmented Generation, which does exactly what its name suggests, helps facilitate generation by retrieving of data.

### Comparing the two AI models

The key difference I noticed is that the nomic-embed-text model doesn't generate responses like the conversational qwen2.5:0.5b, instead it acts more as a search where they convert text into vectors that capture the general meaning of the text.

---

## Building a Personal Knowledge Base

In this step, I'm going to generate the profile text which we will later turn into embeddings. Embeddings are a way of translating human language into lists of numbers that a computer can mathematically compare.

![Image](http://learn.nextwork.org/confident_white_wise_huckleberry/uploads/ai-devops-api_g3h7m2r5)

### Creating the profile document

I included information about my hobbies, fun facts about me, my name, and my career goals.

### Ingesting Data into ChromaDB

Before the API can retrieve answers, the context needs to be converted into vector embeddings and stored. I created a dedicated script (`build_knowledge_base.py`) to handle data ingestion. 

```python
# snippet from build_knowledge_base.py

# split text into chunks separated by a double line break
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# initialize ChromaDB to save data to disk
client = chromadb.PersistentClient(path="./chroma_db")

# connect to the ollama embedding model
my_embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text", 
    url="http://localhost:11434"
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=my_embedding_function
)

# add chunks to the collection with metadata
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source": "profile", "chunk_index": i} for i in range(len(chunks))]
)
```
> **View the full script:** [`build_knowledge_base.py`](./build_knowledge_base.py)

### How semantic search finds relevant chunks

When I ask a question, ChromaDB finds the chunks whose vectors are closest in value to my question's vectors. 

---

## Creating the RAG API with FastAPI

In this step, I'm going to build an API that connects our knowledge base to the model. I'll test it using the built in swagger UI provided by fastAPI

![Image](http://learn.nextwork.org/confident_white_wise_huckleberry/uploads/ai-devops-api_j5m1r8t2)

### How the /ask endpoint works

When a question comes in, my endpoint starts with retrieving the context from our chromaDB, then we move on to augmentation where we feed our prompt with the data retrieved in the first step. Last but not least comes the generation part where we send over our augmented prompt to the model to generate a response.

```python
# snippet from main.py
@app.get("/ask")
def ask(question: str):
    # 1. Retrieval: search chromadb for the 2 most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=2
    )

    # 2. Augmentation: join chunks and build the prompt
    context = "\n\n".join(results["documents"][0])
    
    augmented_prompt = f"""
    Use the following context to answer the question.
    If the context doesn't contain relevant information, say so.
    
    Context:
    {context}
    
    Question: {question}
    """

    # 3. Generation: send augmented prompt to the local LLM
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}]
    )

    # Return the answer along with the context used so users can verify the source
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0]
    }
```
> **View the full application:** [`main.py`](./main.py)

### Testing with Swagger UI

I tested my API by asking it for my name. The AI answered with my name through the context I provided in the chromaDB and sourced the context in its response.

---

## Extending to a Multi-User AI Directory

In this project extension, I'm adding multi-user support because real world RAG systems almost always serve multiple users. Multi-tenancy means multiple users get to use the same application infrastructure and database with each user having their own place in the database.

![Image](http://learn.nextwork.org/confident_white_wise_huckleberry/uploads/ai-devops-api_d5g9k3n7)

### Adding the POST /documents endpoint

In this project extension, I added a POST endpoint that allows the addition of new users alongside their own contexts. Metadata filtering allows the pipeline to filter the contexts retrieved from the knowledge base for our prompt.

![Image](http://learn.nextwork.org/confident_white_wise_huckleberry/uploads/ai-devops-api_r8t2w6y1)

### Verifying multi-user filtering

In this project extension, I tested multi-user queries by asking questions and verifying that the response came back with context sourced from the user asking the query. The filter works because it queries the knowledge base for context with the metadata tags including the username.

```python
# snippet from main.py

# Applying a metadata filter for multi-tenancy based on the user query parameter
@app.get("/ask")
def ask(question: str, user: str = None):
    query_args = {
        "query_texts": [question],
        "n_results": 2
    }

    # dynamically applying the metadata filter
    if user:
        query_args["where"] = {"user_name": user}

    # unpacking arguments to execute the query
    results = collection.query(**query_args)
    
    # ... augmentation and generation ...
```

---

## Wrapping Up

I did this project today to learn how to connect a local LLM to a private vector database using FastAPI and ChromaDB. Another skill I want to learn is how to containerize this AI application with Docker and deploy it using Kubernetes.

---

---
