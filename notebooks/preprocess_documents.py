from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Create a path for the data
path = "skincare_data/"
embeddings_dir = "./embeddings"  # Directory to store embeddings

# Load PDF documents from a specified directory
def load_documents():
    loader = PyPDFDirectoryLoader(path, glob="*.pdf")
    documents = loader.load()
    return documents

# Split a list of documents into smaller chunks using a text splitter
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# Process and store embeddings
def process_and_store_embeddings(documents):
    chunks = split_documents(documents)

    # Create or load the vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        persist_directory=embeddings_dir  # Specify the directory here
    )

    # Persist the vector store to disk
    vector_db.persist()

# Initialize the RAG components
def initialize_rag(persist_directory):
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        )  # Load embeddings from disk

    local_model = "mistral"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate five alternative versions of the user's question to improve 
        document retrieval from a vector database. Separate each version with a newline.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Recommend cosmetics using the precomputed embeddings and RAG
def get_recommendation(skin_type, label_filter, product_ingredient, chain):
    answer = ""
    try:
        query = f"I have {skin_type} skin and want to use a product {label_filter} with these ingredients {product_ingredient}. Is there any active ingredient I should avoid and do you have any user recommendations?"

        try:
            rag_result = chain.invoke({"question": query})
        except Exception as e:
            print(f"Error invoking RAG chain: {e}")
            rag_result = ""

        if rag_result:
            try:
                result = rag_result.split(":")
                answer = result[1].strip()
                print(answer)
                print(type(rag_result))
            except Exception as e:
                print(f"Error processing RAG result: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return answer

# Run the preprocessing step
if __name__ == "__main__":
    documents = load_documents()
    process_and_store_embeddings(documents)
