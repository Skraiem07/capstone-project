# take environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os
api_key = os.environ["OPENAI_API_KEY"]

# Paths
pdf_path = "skincare_data/"  # Directory containing your PDFs
faiss_index_path = "faiss_index_react"  # Directory to store the FAISS index

# Load PDF documents
def load_documents():
    loader = PyPDFDirectoryLoader(pdf_path, glob="*.pdf")
    documents = loader.load()
    return documents

# Split a list of documents into smaller chunks using a text splitter
def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )
    return text_splitter.split_documents(documents=documents)

# Process and store embeddings using FAISS
def process_and_store_embeddings(documents, faiss_index_path):
    chunks = split_documents(documents)
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_index_path)

# Initialize the RAG components using FAISS
def initialize_rag_2(faiss_index_path):
    embeddings = OpenAIEmbeddings()
    
    # Load FAISS vector store
    new_vectorstore = FAISS.load_local(
        faiss_index_path, embeddings, allow_dangerous_deserialization=True
    )

    # Load retrieval QA chat prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create the document chain
    combine_docs_chain = create_stuff_documents_chain(
        ChatOpenAI(), retrieval_qa_chat_prompt
    )

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    return retrieval_chain

# Recommend cosmetics using the precomputed embeddings and RAG
def get_recommendation_2(skin_type, label_filter, product_ingredient, retrieval_chain):
    answer = ""
    try:
        query = f"I have {skin_type} skin and want to use a product {label_filter} with these ingredients {product_ingredient}. Is there any active ingredient I should avoid and do you have any user recommendations?"

        try:
            res = retrieval_chain.invoke({"input": query})
            answer = res['answer'].strip("\n")
            print(answer)
        except Exception as e:
            print(f"Error invoking retrieval chain: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return answer

# Run the preprocessing step
if __name__ == "__main__":
    documents = load_documents()
    process_and_store_embeddings(documents, faiss_index_path)

    retrieval_chain = initialize_rag_2(faiss_index_path)

    # Example of getting a recommendation
    skin_type = "oily"
    label_filter = "anti-aging"
    product_ingredient = "retinol"
    
    recommendation = get_recommendation_2(skin_type, label_filter, product_ingredient, retrieval_chain)
    print(f"Recommendation: {recommendation}")





