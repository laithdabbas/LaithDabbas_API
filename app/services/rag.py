from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import ollama


DATA_PATH = "data"
CHROMA_PATH = "chroma_db"


def build_vector_db():

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(name="documents")

    loader = PyPDFDirectoryLoader(DATA_PATH)

    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(raw_documents)

    documents = [chunk.page_content for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]

    ids = ["ID" + str(i) for i in range(len(chunks))]

    batch_size = 50

    for i in range(0, len(documents), batch_size):
     collection.upsert(
        documents=documents[i:i+batch_size],
        metadatas=metadata[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

    return collection


def retrieve_docs(collection, query):

    results = collection.query(
        query_texts=[query],
        n_results=4
    )

    return results["documents"][0]





def ask_llm(context_docs, question):

    prompt = f"""
     You are a helpful assistant. Answer ONLY using the provided context.

     Context:
     {context_docs}

     Question: {question}
     """

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
def extract_information(context_docs, fields):

    prompt = f"""
    You are an information extraction system.

    Extract ONLY the following fields from the context.
    Return the result as JSON.

    Fields:
    {fields}

    Context:
    {context_docs}
    """

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]