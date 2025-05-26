from transformers import pipeline

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_pipeline = pipeline("sentiment-analysis")

# Define candidate intents
intents = [
    "troubleshooting", "setup help", "connectivity issue", 
    "hardware failure", "feature request", "general question"
]

def classify_query(query):
    sentiment_result = sentiment_pipeline(query)[0]
    sentiment = sentiment_result["label"].lower()

    intent_result = classifier(query, intents)
    intent = intent_result["labels"][0]

    return intent, sentiment
