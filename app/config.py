import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['PINECONE_ENVIRONMENT'] = PINECONE_ENVIRONMENT
os.environ['PINECONE_INDEX_NAME'] = PINECONE_INDEX_NAME

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY