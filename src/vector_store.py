from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="data/chroma")

def get_or_create_collection(collection_name="medical_docs"):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def add_chunks_to_db(chunks, source_name):
    collection = get_or_create_collection()
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": source_name} for _ in chunks]
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")

def search_similar_chunks(query, top_k=3, source_filter=None):
    collection = get_or_create_collection()
    query_embedding = embedding_model.encode([query]).tolist()

    # Filter by source document if specified
    if source_filter:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"source": source_filter}
        )
    else:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

    return results['documents'][0]

def get_all_documents():
    """Return list of all uploaded document names"""
    collection = get_or_create_collection()
    results = collection.get()
    
    if not results['metadatas']:
        return []
    
    # Get unique document names
    sources = list(set([m['source'] for m in results['metadatas']]))
    return sources

def delete_document(source_name):
    """Delete all chunks from a specific document"""
    collection = get_or_create_collection()
    results = collection.get(where={"source": source_name})
    
    if results['ids']:
        collection.delete(ids=results['ids'])
        print(f"Deleted {len(results['ids'])} chunks from {source_name}")