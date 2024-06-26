"""
Embed data and queries with the same embedding function. Two examples are given here.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import ollama


"""
This will work without any additional setup.
"""
def get_hg_embedding_function():
    vectorizer = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return vectorizer

"""
This requires ollama to be running locally.
"""
def get_ollama_embedding_function():
    ollama_emb = ollama.OllamaEmbeddings(model='nomic-embed-text')

    return ollama_emb
