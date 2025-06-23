import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
import re

# --- Configuration ---
# No API key needed for Hugging Face models

# 1. Define the silo
silo_name = "uet_main"
# Use a sitemap or specific starting URLs for this section
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

def crawl_website(start_url, max_depth=3, max_pages=50):
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

# 2. Check if vectorstore already exists
if os.path.exists(vectorstore_path):
    print(f"Loading existing vectorstore from {vectorstore_path}")
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )
    print(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
else:
    print(f"Creating new vectorstore at {vectorstore_path}")
    
    # Crawl the website with better filtering
    all_docs = []
    for start_url in start_urls:
        docs = crawl_website(start_url, max_depth=3, max_pages=50)
        all_docs.extend(docs)
    
    print(f"Found {len(all_docs)} documents from crawling")
    
    if not all_docs:
        print("No documents found. Creating empty vectorstore.")
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embedding_function
        )
    else:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        print(f"Ingesting {len(splits)} chunks for the '{silo_name}' silo.")

        # Create and persist the vector store for this silo
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=vectorstore_path
        )

        print(f"'{silo_name}' vector store created and saved to {vectorstore_path}")