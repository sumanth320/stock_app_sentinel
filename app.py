import os
import finnhub
import chromadb
import logging
import datetime
import time
import requests
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datetime import timedelta

# --- LOGGING CONFIG ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Sentinel")

# --- CLIENT INITIALIZATION ---
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_KEY:
    logger.error("FINNHUB_API_KEY not found in environment variables.")
    exit(1)

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# Persistent ChromaDB Client
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
    logger.info("ChromaDB & Embedding Model Loaded.")
except Exception as e:
    logger.error(f"Initialization Failed: {str(e)}")
    exit(1)


# --- CORE RAG FUNCTIONS ---

def ingest_stock_news(ticker):
    """Phase 1: Ingestion (Data -> Vector DB)"""
    print(f"\n>>> Fetching news for {ticker}...")
    start_date = (datetime.datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    try:
        news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        if not news:
            print(f"⚠️ No recent news found for {ticker}.")
            return

        # Process top 15 most recent articles
        items = news[:15]
        for art in items:
            collection.upsert(
                documents=[f"{art['headline']}. {art['summary']}"],
                metadatas=[{
                    "ticker": ticker.upper(),
                    "timestamp": int(art['datetime']),
                    "url": art.get('url', '')
                }],
                ids=[f"{ticker}_{art['datetime']}"]
            )
        print(f"✅ Ingest complete: {len(items)} articles added for {ticker}.")
    except Exception as e:
        print(f"❌ Ingest failed: {str(e)}")


def generate_answer(question, context_list):
    """Phase 3 & 4: Augmentation & Generation (Docker-to-Docker)"""
    if not context_list:
        return "I couldn't find any relevant news in my database to answer that."

    context_text = "\n---\n".join(context_list)
    prompt = f"""
    You are a professional financial analyst. 
    Use the following news snippets to answer the user's question.
    If the answer isn't in the context, say you don't have enough data.

    CONTEXT:
    {context_text}

    USER QUESTION: {question}

    ANALYST RESPONSE:"""

    try:
        # 180s timeout is necessary for CPU-based inference on 8B models
        response = requests.post(
            "http://host.docker.internal:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False  # Ensure streaming is off to avoid JSON errors
            },
            timeout=180
        )

        # Check for HTTP errors (404, 500, etc)
        response.raise_for_status()

        # Handle potential "Extra data" by accessing the first JSON object
        try:
            return response.json().get('response', 'Error: No response field found.')
        except ValueError:
            # If Ollama sends multiple JSON objects, we split and take the first valid one
            raw_text = response.text.strip().split('\n')[0]
            import json
            return json.loads(raw_text).get('response', 'Error: Failed to parse LLM JSON.')

    except requests.exceptions.HTTPError as e:
        return f"LLM Server Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Local LLM Error: {str(e)}"


def query_sentinel(question, ticker_filter=None, days_back=7):
    """Phase 2: Retrieval (Hybrid Metadata + Semantic Search)"""
    limit_ts = int((datetime.datetime.now() - timedelta(days=days_back)).timestamp())

    # Metadata Filtering Logic
    if ticker_filter:
        where_clause = {
            "$and": [
                {"timestamp": {"$gte": limit_ts}},
                {"ticker": ticker_filter.upper()}
            ]
        }
    else:
        where_clause = {"timestamp": {"$gte": limit_ts}}

    try:
        results = collection.query(
            query_texts=[question],
            n_results=5,
            where=where_clause
        )

        if results['documents'] and results['documents'][0]:
            context_list = results['documents'][0]
            # Send context to the LLM
            answer = generate_answer(question, context_list)
            print(f"\n🤖 SENTINEL AI ANALYSIS:\n{answer}")
        else:
            print(f"\n🔍 No news found for that timeframe/ticker in the database.")
    except Exception as e:
        print(f"❌ Search Error: {str(e)}")


def show_stats():
    """Diagnostic helper for database health"""
    results = collection.get()
    ids = results.get('ids', [])
    metadatas = results.get('metadatas', [])

    tickers = set(m.get('ticker') for m in metadatas if m)

    print("\n" + "=" * 40)
    print(f"📊 SENTINEL DATABASE STATUS")
    print(f"Total Documents: {len(ids)}")
    print(f"Unique Tickers:  {', '.join(tickers) if tickers else 'None'}")
    print("=" * 40 + "\n")


# --- MAIN INTERACTIVE LOOP ---

if __name__ == "__main__":
    print("\n" + "*" * 30)
    print("SENTINEL INTELLIGENCE SYSTEM ACTIVE")
    print("*" * 30)

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