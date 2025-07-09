from langchain_ollama import OllamaEmbeddings

def embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")