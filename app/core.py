import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .loader import load_and_split_pdf
from .config import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

def load_and_split_markdown(file_path):
    """Load and split markdown file for embeddings"""
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    split_documents = text_splitter.split_documents(documents)
    print(f"Split markdown into {len(split_documents)} chunks")
    return split_documents

def initialize_chain():
    # Initialize Google LLM and embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists in the list of indexes
    index_list = [index.name for index in pc.list_indexes()]
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in index_list:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        index_empty = True
    else:
        # Check if index is empty
        index_stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
        index_empty = index_stats['total_vector_count'] == 0

    # Initialize PineconeVectorStore
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )

    # Only add documents if the index was just created or is empty
    if index_empty:
        print("Loading and embedding documents into Pinecone index...")
        
        handbook_docs = load_and_split_pdf("Handbook_Travelling_Allowances_2023.pdf")
        faq_docs = load_and_split_markdown("pcda_FAQ.md")
        all_docs = handbook_docs + faq_docs

        # Batch documents to avoid exceeding Pinecone's request size limit
        batch_size = 50
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            try:
                vectorstore.add_documents(batch)
                print(f"Processed batch {i//batch_size + 1}/{(len(all_docs)//batch_size)+1}")
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                smaller_batch = batch[:len(batch)//2]
                vectorstore.add_documents(smaller_batch)

        print("Documents embedded and added to Pinecone index.")
    else:
        print("Using existing documents in Pinecone index.")

    # Setup retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "score_threshold": 0.65}
    )

    return llm, retriever