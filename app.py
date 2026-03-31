import os, finnhub, chromadb, logging, json, datetime
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datetime import timedelta


# --- STRUCTURED LOGGING ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        return json.dumps(log_record)


logger = logging.getLogger("Sentinel")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- CONFIG & CLIENTS ---
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_KEY:
    logger.error("API Key Missing")
    exit(1)

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
client = chromadb.PersistentClient(path="./chroma_data", settings=Settings(anonymized_telemetry=False))
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="stock_news", embedding_function=embedding_func)


def ingest_stock_news(ticker):
    logger.info(f"Starting fetch", extra={"ticker": ticker, "action": "ingest_start"})
    start_date = (datetime.datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    try:
        news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        count = len(news[:15])
        for i, art in enumerate(news[:15]):
            collection.upsert(
                documents=[f"{art['headline']}. {art['summary']}"],
                metadatas=[{
                    "ticker": ticker,
                    "timestamp": int(art['datetime'])  # Finnhub gives Unix timestamps
                }],
                ids=[f"{ticker}_{art['datetime']}"]
            )
        logger.info("Ingest complete", extra={"ticker": ticker, "count": count, "status": "success"})
    except Exception as e:
        logger.error("Ingest failed", extra={"ticker": ticker, "error": str(e)})


def query_sentinel(question):
    logger.info("New query received", extra={"action": "query_start"})
    try:
        start_time = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp())

        results = collection.query(
            query_texts=[question],
            n_results=4,
            # Metadata Filter: Only show news from the last X days
            where={"timestamp": {"$gte": start_time}}
        )

        # Log the success so Grafana can see it
        logger.info("Query successful", extra={
            "action": "query_end",
            "result_count": len(results['documents'][0])
        })

        print("\n--- SENTINEL FINDINGS ---")
        for i in range(len(results['documents'][0])):
            print(f"\nSource {i + 1}: {results['metadatas'][0][i]['url']}")
            print(f"Content: {results['documents'][0][i][:200]}...")
        print("\n-------------------------\n")

    except Exception as e:
        logger.error("Query failed", extra={"error": str(e)})

if __name__ == "__main__":
    while True:
        cmd = input("[fetch/ask/exit]: ").lower()
        if cmd == 'fetch':
            ingest_stock_news(input("Ticker: ").upper())
        elif cmd == 'ask':
            query_sentinel(input("Your Question: "))
        elif cmd == 'exit': break