import os
#os.environ["OPENAI_API_KEY"]=""
os.environ["OPENAI_API_KEY"]=""
import os
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --------------------------
# CONFIG
# --------------------------
DB_FAISS_PATH = 'vectorstore/sales_db_faiss_1'
PDF_FOLDER = r"/home/ubuntu/genai_testing/Sales_ref_documents"

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    all_docs = []

    # Loop through all PDF files in folder
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Loading PDF: {filename}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

    print(f"✅ Total documents loaded: {len(all_docs)}")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    print(f"✅ Total text chunks: {len(splits)}")

    # Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings(chunk_size=100)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Save FAISS index locally
    vectorstore.save_local(DB_FAISS_PATH)
    print(f"🎉 FAISS vectorstore saved successfully at {DB_FAISS_PATH}")
