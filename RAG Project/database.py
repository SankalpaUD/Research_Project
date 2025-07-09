import os, shutil, argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embedding import embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data/university_handbook.pdf"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    documents = PyMuPDFLoader(DATA_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    chunks = add_chunk_ids(chunks)

    print("\nPreviewing first 5 chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content[:500])

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())
    existing_ids = set(db.get(include=[]).get("ids", []))
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if new_chunks:
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        print(f"Added {len(new_chunks)} new chunks.")
    else:
        print("No new documents to add.")

def add_chunk_ids(chunks):
    last = None
    index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        key = f"{source}:{page}"
        index = index + 1 if key == last else 0
        chunk.metadata["id"] = f"{key}:{index}"
        last = key
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()