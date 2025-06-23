import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.documents import Document
import re
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Set your Voyage API key in .env file: VOYAGE_API_KEY=your_key_here

# 1. Define the silo
silo_name = "uet_main"
start_urls = ["https://www.uetmardan.edu.pk/uetm/"]
vectorstore_path = f"./vectorstores/{silo_name}"

def is_valid_url(url):
    """Check if URL is valid and not a file download"""
    parsed = urlparse(url)
    
    # Skip common file extensions
    file_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.doc', '.docx', 
                      '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.exe']
    
    for ext in file_extensions:
        if url.lower().endswith(ext):
            return False
    
    # Only process HTTP/HTTPS URLs
    return parsed.scheme in ['http', 'https']

def extract_text_from_html(html_content):
    """Extract clean text from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    # Get text and clean it up
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def crawl_website(start_url, max_depth=2, max_pages=20):  # Reduced limits
    """Crawl website and extract text content"""
    visited = set()
    to_visit = [(start_url, 0)]  # (url, depth)
    documents = []
    
    while to_visit and len(documents) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        if current_url in visited or depth > max_depth:
            continue
            
        visited.add(current_url)
        
        try:
            print(f"Crawling: {current_url} (depth: {depth})")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check if it's HTML content
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                print(f"Skipping non-HTML content: {content_type}")
                continue
            
            # Extract text from HTML
            text = extract_text_from_html(response.text)
            
            if len(text.strip()) > 100:  # Only keep documents with substantial content
                doc = Document(
                    page_content=text,
                    metadata={"source": current_url, "depth": depth}
                )
                documents.append(doc)
                print(f"Added document with {len(text)} characters")
            
            # Find links for next level
            if depth < max_depth:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']  # type: ignore
                    absolute_url = urljoin(current_url, str(href))
                    
                    # Only process URLs from the same domain
                    if (urlparse(absolute_url).netloc == urlparse(start_url).netloc and 
                        is_valid_url(absolute_url) and 
                        absolute_url not in visited):
                        to_visit.append((absolute_url, depth + 1))
                        
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            continue
    
    return documents

class RateLimitedVoyageEmbeddings:
    """Wrapper for VoyageAI embeddings with rate limiting"""
    
    def __init__(self, api_key, model="voyage-3.5-lite", requests_per_minute=3):
        self.embeddings = VoyageAIEmbeddings(
            voyage_api_key=api_key,
            model=model,
            batch_size=1  # Process one at a time to avoid rate limits
        )
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = 0
        
    def embed_documents(self, texts):
        """Embed documents with rate limiting"""
        embeddings = []
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            
            try:
                print(f"Embedding document {i+1}/{total_texts}...")
                # Truncate text to avoid token limits (roughly 4 chars per token)
                truncated_text = text[:4000]  # ~1000 tokens max
                embedding = self.embeddings.embed_documents([truncated_text])
                embeddings.extend(embedding)
                self.last_request_time = time.time()
                
            except Exception as e:
                print(f"Error embedding document {i+1}: {e}")
                if "rate limit" in str(e).lower():
                    print("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    # Retry once
                    try:
                        embedding = self.embeddings.embed_documents([truncated_text])
                        embeddings.extend(embedding)
                        self.last_request_time = time.time()
                    except Exception as retry_e:
                        print(f"Retry failed: {retry_e}")
                        # Use a zero vector as fallback
                        embeddings.append([0.0] * 1024)
                else:
                    # Use a zero vector as fallback
                    embeddings.append([0.0] * 1024)
        
        return embeddings
    
    def embed_query(self, text):
        """Embed query with rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        
        try:
            # Truncate text to avoid token limits
            truncated_text = text[:4000]  # ~1000 tokens max
            embedding = self.embeddings.embed_query(truncated_text)
            self.last_request_time = time.time()
            return embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            if "rate limit" in str(e).lower():
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                try:
                    embedding = self.embeddings.embed_query(truncated_text)
                    self.last_request_time = time.time()
                    return embedding
                except Exception as retry_e:
                    print(f"Retry failed: {retry_e}")
                    return [0.0] * 1024
            else:
                return [0.0] * 1024

def get_embedding_function():
    """Get rate-limited Voyage AI embedding function"""
    return RateLimitedVoyageEmbeddings(
        api_key=os.getenv("VOYAGE_API_KEY"),
        model="voyage-3.5-lite",  # Cheapest option with 200M free tokens
        requests_per_minute=3  # Conservative rate limit (free tier is 3 RPM)
    )

# 2. Check if vectorstore already exists
if os.path.exists(vectorstore_path):
    print(f"Loading existing vectorstore from {vectorstore_path}")
    embedding_function = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )
    print(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
else:
    print(f"Creating new vectorstore at {vectorstore_path}")
    
    # Crawl the website with reduced limits to stay within free tier
    all_docs = []
    for start_url in start_urls:
        docs = crawl_website(start_url, max_depth=2, max_pages=20)  # Reduced limits
        all_docs.extend(docs)
    
    print(f"Found {len(all_docs)} documents from crawling")
    
    if not all_docs:
        print("No documents found. Creating empty vectorstore.")
        embedding_function = get_embedding_function()
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embedding_function
        )
    else:
        # Split the documents into smaller chunks to stay within token limits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks (~250 tokens each)
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(all_docs)
        
        # Limit total chunks to stay within free tier
        max_chunks = 50  # Conservative limit
        if len(splits) > max_chunks:
            print(f"Limiting to {max_chunks} chunks to stay within free tier")
            splits = splits[:max_chunks]

        print(f"Ingesting {len(splits)} chunks for the '{silo_name}' silo.")
        print("This will take some time due to rate limiting...")

        # Create and persist the vector store for this silo
        embedding_function = get_embedding_function()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=vectorstore_path
        )

        print(f"'{silo_name}' vector store created and saved to {vectorstore_path}")
