import yfinance as yf
import chromadb
from sentence_transformers import SentenceTransformer
import datetime

# --- CONFIGURATION ---
DB_PATH = "./chroma_data"
MODEL_NAME = "all-MiniLM-L6-v2"

# 1. Initialize Tools
print("--- Initializing AI Model and Database ---")
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="stock_news_v1")


def ingest_stock_news(ticker_symbol):
    print(f"\n[1/3] Searching for {ticker_symbol} news...")

    # 2026 Recommended Method: Use yf.Search for better results
    search = yf.Search(ticker_symbol, news_count=10)
    news_items = search.news

    if not news_items:
        print(f"!!! Yahoo returned NO news for {ticker_symbol}. Try again in a minute.")
        return

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(news_items):
        # Search returns a slightly different structure
        title = item.get('title', 'No Title')
        publisher = item.get('publisher', 'Unknown Source')
        link = item.get('link', 'No Link')

        # We combine them for a rich context
        content_str = f"Source: {publisher}. Title: {title}"
        doc_id = f"{ticker_symbol}_{i}_{datetime.datetime.now().timestamp()}"

        documents.append(content_str)
        metadatas.append({"ticker": ticker_symbol, "link": link})
        ids.append(doc_id)

    print(f"[2/3] Embedding {len(documents)} articles...")
    embeddings = model.encode(documents).tolist()

    print(f"[3/3] Saving to ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"--- SUCCESS: {len(documents)} articles for {ticker_symbol} stored ---")

def query_rag(user_question):
    """Searches the database for the answer."""
    query_vector = model.encode([user_question]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=2
    )

    print("\n--- TOP RELEVANT NEWS FOUND ---")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i + 1}:")
        print(doc)
        print(f"Source: {results['metadatas'][0][i]['link']}")


# --- MAIN LOOP ---
if __name__ == "__main__":
    while True:
        action = input("\nEnter 'fetch' to get new data, 'ask' to query, or 'exit': ").lower()

        if action == 'fetch':
            symbol = input("Enter ticker (e.g., NVDA, AAPL, TSLA): ").upper()
            ingest_stock_news(symbol)
        elif action == 'ask':
            q = input("What do you want to know about your stocks? ")
            query_rag(q)
        elif action == 'exit':
            break