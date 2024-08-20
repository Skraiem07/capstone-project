from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever



#create a path for the data 
path="skincare_data/"

def load_documents():
    """
    Load PDF documents from a specified directory.

    Returns:
    list of Document: A list of documents loaded from the PDF files in the directory.
    """
    
    # Create an instance of PyPDFDirectoryLoader to load PDF files from the specified directory.
    # 'path' specifies the directory where PDF files are located.
    # 'glob="*.pdf"' ensures that only files with a .pdf extension are loaded.
    loader = PyPDFDirectoryLoader(path, glob="*.pdf")
    
    # Use the loader to load all PDF documents from the specified directory.
    # The 'load' method returns a list of Document objects.
    documents = loader.load()
    
    # Return the list of loaded documents.
    return documents



documents = load_documents()
print(f"Number of documents loaded: {len(documents)}")


def split_documents(documents):
    """
    Splits a list of documents into smaller chunks using a text splitter.
    
    Parameters:
    documents (list of str): The list of documents to be split into chunks.
    
    Returns:
    list of str: A list of text chunks resulting from the split operation.
    """
    
    # Create an instance of RecursiveCharacterTextSplitter for splitting documents.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Size of each chunk in characters
        chunk_overlap=500,        # Number of overlapping characters between chunks
        length_function=len,      # Function to determine the length of the text (using len here)
        is_separator_regex=False  # Indicates that separators (such as spaces or punctuation) should NOT be treated as regular expressions
    )
    
    # Split the documents into chunks and return the result.
    return text_splitter.split_documents(documents)


chunks = split_documents(documents)

# Create or update a vector database with embeddings for the provided documents.
# The 'vector_db' will store document embeddings which are useful for similarity searches and retrieval.

vector_db = Chroma.from_documents(
    documents=chunks,  # List of document chunks to be added to the vector database
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),  # Embedding model used to convert documents into vector representations
    collection_name="local-rag"  # The name given to the collection within the vector database. This helps organize and identify the specific set of document embeddings stored in the database.
)


# Define the name of the local model to be used for the language model.
local_model = "mistral"  # "mistral" refers to the specific local model (with 7 billion parameters) you want to use.

# Initialize a ChatOllama instance using the specified local model.
# This sets up the large language model (LLM) that will be used for generating responses in a chat-based application.
llm = ChatOllama(model=local_model)  # Create the LLM instance with the "mistral" model.

# Define a prompt template for generating alternative versions of a user's question.
# This is used to enhance document retrieval from a vector database by providing
# multiple variations of the question, thereby improving the chances of retrieving relevant documents.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Generate five alternative versions of the user's question to improve 
    document retrieval from a vector database. Separate each version with a newline.
    Original question: {question}"""
)


# Initialize a MultiQueryRetriever instance to generate more than five queries
# The retriever uses a large language model (LLM) for generating queries.
# It is configured with a vector database retriever and a custom prompt template (QUERY_PROMPT).
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
#new Rag prompt taking input from prototype cosine similarity 

prompt = ChatPromptTemplate.from_template(template)

# Create a processing chain to handle a question-answering task using a sequence of operations.

chain = (
    # Step 1: Retrieve the relevant context based on the user's question.
    {"context": retriever, "question": RunnablePassthrough()}  # The retriever gets context; question passes through unchanged.

    # Step 2: Use the prompt template to format the context and question into a structured prompt for the LLM.
    | prompt  # The formatted prompt is prepared using the provided context and question.

    # Step 3: Pass the prompt to the large language model (LLM) to generate a response.
    | llm  # The LLM generates an answer based on the prompt.

    # Step 4: Parse the output from the LLM into a string format that can be easily used or displayed.
    | StrOutputParser()  # The output from the LLM is parsed into a final string format.
)




# Recommend cosmetics using both the embedding model and RAG
"""def get_recommendation(skin_type, label_filter, product_ingredient):
    try:
        # Use RAG to enhance the recommendation
        query = f"I have {skin_type} skin and want to use a product {label_filter} with these ingredients {product_ingredient}. Is there any active ingredient I should avoid and do you have any user recommendations?"
        
        rag_result = chain.invoke({"question": query})
        
        # Split the result to get the answer part
        result = rag_result.split(":")
        answer = result[1].strip()  # Strip any leading/trailing whitespace
        
        print(answer)
        print(type(rag_result))
        
        return answer

    except Exception as e:
        # Log the error if necessary
        print(f"An error occurred: {e}")
        # Return an empty string in case of error
        return """

def get_recommendation(skin_type, label_filter, product_ingredient):
    answer = ""  # Initialize the answer as an empty string
    try:
        # Step 1: Construct the query
        query = f"I have {skin_type} skin and want to use a product {label_filter} with these ingredients {product_ingredient}. Is there any active ingredient I should avoid and do you have any user recommendations?"

        # Step 2: Use RAG to enhance the recommendation
        try:
            rag_result = chain.invoke({"question": query})
        except Exception as e:
            print(f"Error invoking RAG chain: {e}")
            rag_result = ""

        # Step 3: Process the result if available
        if rag_result:
            try:
                result = rag_result.split(":")
                answer = result[1].strip()  # Strip any leading/trailing whitespace
                print(answer)
                print(type(rag_result))
            except Exception as e:
                print(f"Error processing RAG result: {e}")

    except Exception as e:
        # Log any unexpected errors that might occur
        print(f"An unexpected error occurred: {e}")
    
    # Return the answer (could be empty if any errors occurred)
    return answer
