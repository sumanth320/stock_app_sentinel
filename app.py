import os
import finnhub
import chromadb
import logging
import datetime
import hashlib
import time
import requests
import json
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datetime import timedelta
from dotenv import load_dotenv

# --- INITIALIZATION & CONFIG ---
load_dotenv()  # Loads from .env file for K8s-ready configuration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Sentinel")

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
# Reaches out to Mac GPU via host.docker.internal
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")

if not FINNHUB_KEY:
    logger.error("FINNHUB_API_KEY not found. Please check your .env file.")
    exit(1)

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# 1. PERSISTENCE: Data now saves to ./chroma_data mapped folder
client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(anonymized_telemetry=False)
)

# Initialize Embedding Function (Local SentenceTransformer)
try:
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="stock_news",
        embedding_function=embedding_func
    )
    logger.info("Sentinel Vector Store initialized with Persistence.")
except Exception as e:
    logger.error(f"Initialization Failed: {str(e)}")
    exit(1)


def fetch_stock_news(ticker):
    """Retrieves raw news from Finnhub API."""
    import datetime
    # Define time window (e.g., last 3 days)
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y-%m-%d')

    try:
        # Use the global finnhub_client we initialized earlier
        news = finnhub_client.company_news(ticker.upper(), _from=from_date, to=to_date)
        return news
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return []


# --- CORE RAG FUNCTIONS ---

def ingest_stock_news(ticker):
    print(f"📡 Fetching latest news for {ticker}...")
    news = fetch_stock_news(ticker)

    if not news:
        return

    # 1. Get IDs of existing documents in this collection
    # This prevents us from re-embedding what we already have
    existing_data = collection.get()
    existing_ids = set(existing_data['ids'])

    new_documents = []
    new_metadatas = []
    new_ids = []

    for article in news:
        # Create a unique ID based on the URL or the Finnhub ID
        # Using a hash ensures the ID is always the same for the same content
        article_url = article.get('url', '')
        unique_id = hashlib.md5(article_url.encode()).hexdigest()

        # 2. THE DEDUPLICATION CHECK
        if unique_id not in existing_ids:
            new_documents.append(article['summary'])
            new_metadatas.append({
                "ticker": ticker.upper(),
                "timestamp": article['datetime'],
                "source": article['source']
            })
            new_ids.append(unique_id)

    # 3. Only Batch-Add if there is actually new data
    if new_documents:
        collection.add(
            documents=new_documents,
            metadatas=new_metadatas,
            ids=new_ids
        )
        print(f"✅ Successfully added {len(new_documents)} NEW articles for {ticker}.")
    else:
        print(f"ℹ️ No new articles found for {ticker}. Database is up to date.")


def generate_answer(question, context_list):
    """Phase 3 & 4: Context-Aware Generation"""
    if not context_list:
        return "I couldn't find any relevant news in my database to answer that."

    context_text = "\n---\n".join(context_list)
    prompt = f"""
    You are a professional financial analyst. 
    Use the following news snippets to answer the user's question accurately.
    Base your answer ONLY on the provided context.
    If the information isn't there, state that you don't have enough data.

    CONTEXT:
    {context_text}

    USER QUESTION: {question}

    ANALYST RESPONSE:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False  # Disable streaming for a single clean JSON response
            },
            timeout=180
        )
        response.raise_for_status()

        # 4. ROBUST JSON PARSING: Handles the 'Extra Data' issue
        try:
            return response.json().get('response', 'Error: No response field found.')
        except json.JSONDecodeError:
            # Fallback: Clean the text if heartbeat chunks were included
            lines = response.text.strip().split('\n')
            for line in lines:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        return data['response']
                except:
                    continue
            return "Error: Could not parse LLM response chunks."

    except Exception as e:
        return f"Local LLM Error: {str(e)}"


def hybrid_search(question, ticker_filter=None, n_results=5):
    """
    Combines Semantic Search and Keyword Matching.
    """
    # 1. Semantic Search (The current way)
    limit_ts = int((datetime.datetime.now() - timedelta(days=30)).timestamp())

    where_clause = {"timestamp": {"$gte": limit_ts}}
    if ticker_filter:
        where_clause = {
            "$and": [
                {"timestamp": {"$gte": limit_ts}},
                {"ticker": ticker_filter.upper()}
            ]
        }

    # Semantic Results
    vector_results = collection.query(
        query_texts=[question],
        n_results=n_results * 2,  # Fetch more to allow for filtering/merging
        where=where_clause
    )

    # 2. Simple Keyword "Boost"
    # We look for exact ticker matches or keywords in the results
    final_docs = []
    seen_ids = set()

    # Priority 1: Semantic results that also contain the ticker name
    if ticker_filter:
        ticker_lower = ticker_filter.lower()
        for i, doc in enumerate(vector_results['documents'][0]):
            if ticker_lower in doc.lower():
                final_docs.append(doc)
                seen_ids.add(vector_results['ids'][0][i])

    # Priority 2: Fill the rest with general semantic results
    for i, doc in enumerate(vector_results['documents'][0]):
        doc_id = vector_results['ids'][0][i]
        if doc_id not in seen_ids and len(final_docs) < n_results:
            final_docs.append(doc)
            seen_ids.add(doc_id)

    return final_docs


def query_sentinel(question, ticker_filter=None, days_back=7):
    """
    Phase 2: Hybrid Retrieval (Semantic + Keyword)
    Combines vector similarity with exact keyword matching for higher precision.
    """
    try:
        # Calculate the look-back window
        limit_ts = int((datetime.datetime.now() - timedelta(days=days_back)).timestamp())

        # 1. Define the Metadata Filter (Where Clause)
        if ticker_filter:
            where_clause = {
                "$and": [
                    {"timestamp": {"$gte": limit_ts}},
                    {"ticker": ticker_filter.upper()}
                ]
            }
        else:
            where_clause = {"timestamp": {"$gte": limit_ts}}

        # 2. Execute Vector Search
        # We fetch slightly more results (n=10) to allow our hybrid logic
        # to re-rank and filter them down to the best 5.
        results = collection.query(
            query_texts=[question],
            n_results=10,
            where=where_clause
        )

        if not results['documents'] or not results['documents'][0]:
            print(f"\n🔍 No relevant news found for {ticker_filter or 'all tickers'} in the last {days_back} days.")
            return

        # 3. Hybrid Re-ranking Logic (Simple Reciprocal Rank Fusion)
        # We prioritize documents that contain the ticker or specific keywords
        raw_docs = results['documents'][0]
        refined_context = []

        # Keyword Boost: If a ticker filter is provided, move exact matches to the top
        if ticker_filter:
            target = ticker_filter.upper()
            # First pass: Grab docs that explicitly mention the ticker
            for doc in raw_docs:
                if target in doc.upper():
                    refined_context.append(doc)

            # Second pass: Fill the rest with semantic results that didn't have the keyword
            for doc in raw_docs:
                if doc not in refined_context:
                    refined_context.append(doc)
        else:
            refined_context = raw_docs

        # Limit to the top 5 most relevant snippets for the LLM
        final_context = refined_context[:5]

        # 4. Generate Answer
        print(f"🧠 Analyzing {len(final_context)} relevant snippets...")
        answer = generate_answer(question, final_context)

        print(f"\n🤖 SENTINEL HYBRID AI ANALYSIS:")
        print("-" * 30)
        print(answer)
        print("-" * 30)

    except Exception as e:
        # Catching errors here prevents the whole interactive loop from crashing
        logger.error(f"Search Error: {str(e)}")
        print(f"❌ Sentinel encountered an error during retrieval: {str(e)}")


def show_stats():
    """Diagnostic Helper"""
    results = collection.get()
    ids = results.get('ids', [])
    metadatas = results.get('metadatas', [])
    tickers = set(m.get('ticker') for m in metadatas if m)

    print("\n" + "=" * 45)
    print(f"📊 SENTINEL PERSISTENT DATABASE STATUS")
    print(f"Total Unique Articles: {len(ids)}")
    print(f"Active Tickers:        {', '.join(tickers) if tickers else 'None'}")
    print(f"Storage Path:          ./chroma_data")
    print("=" * 45 + "\n")


# --- MAIN LOOP ---

if __name__ == "__main__":
    print("\n" + "*" * 40)
    print("  SENTINEL RAG SYSTEM : M4-GPU EDITION")
    print("*" * 40)

    while True:
        try:
            cmd = input("\n[fetch / ask / stats / exit]: ").lower().strip()

            if cmd == 'fetch':
                t = input("Enter Stock Ticker (e.g., MU): ").upper().strip()
                if t: ingest_stock_news(t)

            elif cmd == 'stats':
                show_stats()

            elif cmd == 'ask':
                q = input("What would you like to know? ")
                t = input("Filter by Ticker (Enter for all): ").strip()
                d = input("Search window in days (default 7): ").strip()
                query_sentinel(
                    q,
                    ticker_filter=t if t else None,
                    days_back=int(d) if d else 7
                )

            elif cmd == 'exit':
                print("Shutting down Sentinel...")
                break

        except (EOFError, KeyboardInterrupt):
            # Keeps the container alive if no terminal is attached
            time.sleep(1)