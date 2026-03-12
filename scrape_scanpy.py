import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import chromadb
import uuid
import re
from tqdm import tqdm

# The directory our Qwen RAG Agent expects to find the database
CHROMA_DB_DIR = "./scanpy_chroma_db"

# Core scanpy API index pages
API_INDEX_URLS = [
    "https://scanpy.readthedocs.io/en/stable/api/preprocessing.html",
    "https://scanpy.readthedocs.io/en/stable/api/tools.html",
    "https://scanpy.readthedocs.io/en/stable/api/plotting.html",
    "https://scanpy.readthedocs.io/en/stable/api/classes.html"
]

def get_function_links(base_url):
    """Finds all links to generated API documentation from an index page."""
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {base_url}: {e}")
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()
    
    # Sphinx autosummary tables usually point to individual function/class pages
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'scanpy.' in href and '.html' in href:
            full_url = urljoin(base_url, href)
            # Ensure we are staying within the scanpy docs
            if "scanpy.readthedocs.io" in full_url:
                links.add(full_url)
                
    return list(links)

def scrape_function_docs(url):
    """Scrapes the function signature and description from a Sphinx HTML page."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    docs = []
    
    # Sphinx docs wrap Python API entries in <dl class="py function/class/method">
    for dl in soup.find_all('dl', class_=re.compile(r'py (function|class|method)')):
        dt = dl.find('dt')
        if not dt:
            continue
            
        # Clean up the signature (e.g., sc.pp.filter_cells(adata, ...))
        signature = dt.get_text(separator=' ', strip=True)
        signature = signature.replace('[source]', '').strip()
        
        # Find the description block
        dd = dl.find('dd')
        if not dd:
            continue
            
        # Extract the first few paragraphs to capture the summary without bloating LLM context
        paragraphs = dd.find_all('p', recursive=False)
        desc_text = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])
        
        # Fallback if there are no <p> tags
        if not desc_text:
            desc_text = dd.get_text(strip=True)[:500] + "..."
            
        doc_text = f"Signature: {signature}\nDescription: {desc_text}\nSource: {url}"
        docs.append(doc_text)
        
    return docs

def build_scanpy_knowledge_base():
    print("Gathering function URLs from Scanpy API indices...")
    all_function_urls = set()
    for index_url in API_INDEX_URLS:
        urls = get_function_links(index_url)
        all_function_urls.update(urls)
        
    print(f"Found {len(all_function_urls)} unique documentation pages to scrape.")
    
    all_documents = []
    print("Scraping documentation pages...")
    for url in tqdm(all_function_urls, desc="Pages Scraped"):
        docs = scrape_function_docs(url)
        all_documents.extend(docs)
        
    print(f"Successfully scraped {len(all_documents)} function/class definitions.")
    
    print(f"Initializing ChromaDB at '{CHROMA_DB_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Reset the collection if you run this script multiple times
    try:
        client.delete_collection("scanpy_knowledge")
    except ValueError:
        pass # Collection didn't exist yet
        
    collection = client.create_collection(name="scanpy_knowledge")
    
    print("Populating vector database...")
    ids = [str(uuid.uuid4()) for _ in range(len(all_documents))]
    
    # Add to ChromaDB in batches to prevent SQLite payload limits
    batch_size = 100
    for i in tqdm(range(0, len(all_documents), batch_size), desc="DB Ingestion"):
        batch_docs = all_documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(
            documents=batch_docs,
            ids=batch_ids
        )
        
    print("✅ Successfully built the Scanpy RAG database!")

if __name__ == "__main__":
    build_scanpy_knowledge_base()