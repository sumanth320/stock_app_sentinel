import os
import time
import finnhub
import chromadb
from datetime import datetime, timedelta
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# --- 1. CONFIGURATION & SECURITY ---
# Grab API Key from the Environment (Injected via --env-file .env)
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")

if not FINNHUB_KEY:
    print("❌ ERROR: FINNHUB_API_KEY not found! Check your .env file.")
    exit(1)

# Initialize Finnhub Client
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# Initialize ChromaDB with Telemetry Disabled (Fixes the capture() error)
DB_PATH = "./chroma_data"
client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Using Sentence Transformers for high-quality local embeddings
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="stock_sentinel_news",
    embedding_function=embedding_func
)


# --- 2. INGESTION ENGINE ---
def ingest_stock_news(ticker_symbol):
    print(f"\n--- 🛰️ Sentinel Ingest: {ticker_symbol} ---")

    # 60-Day Lookback for better "War Phase" context
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    try:
        # Fetching news (Up to 1 year back is allowed on free tier)
        news_list = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)

        if not news_list:
            print(f"⚠️ No news found for {ticker_symbol} in the last 60 days.")
            return

        print(f"📥 Found {len(news_list)} articles. Storing top 15...")

        for i, article in enumerate(news_list[:15]):
            headline = article.get('headline', 'No Title')
            summary = article.get('summary', '')
            url = article.get('url', '')
            timestamp = article.get('datetime', int(time.time()))

            # Combine Headline + Summary for the RAG Context
            full_text = f"{headline}. {summary}"

            # Create a unique ID to prevent duplicates
            doc_id = f"{ticker_symbol}_{timestamp}_{i}"

            collection.upsert(
                documents=[full_text],
                metadatas=[{
                    "ticker": ticker_symbol,
                    "url": url,
                    "timestamp": timestamp,
                    "date": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                }],
                ids=[doc_id]
            )

        print(f"✅ Successfully updated {ticker_symbol} database.")

    except Exception as e:
        print(f"❌ Finnhub API Error: {e}")


# --- 3. RAG QUERY ENGINE ---
def query_sentinel(user_query):
    print(f"\n--- 🧠 Sentinel Analysis: '{user_query}' ---")

    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )

    if not results['documents'][0]:
        print("🤷 No relevant news found in local database. Try 'fetch' first.")
        return

    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        print(f"\n[{i + 1}] Source Date: {meta['date']}")
        print(f"Insight: {doc}")
        print(f"Read more: {meta['url']}")


# --- 4. MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Stock App Sentinel Initialized (M4 Optimized)")
    while True:
        mode = input("\n[fetch] get news | [ask] query AI | [exit]: ").lower()

        if mode == 'fetch':
            t = input("Enter Ticker: ").upper()
            ingest_stock_news(t)
        elif mode == 'ask':
            q = input("What would you like to know?: ")
            query_sentinel(q)
        elif mode == 'exit':
            print("Shutting down...")
            break