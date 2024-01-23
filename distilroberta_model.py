from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Function to train and save the model (not needed for pre-trained models)
# def train_and_save_model():
    # Code for training if needed
# train_and_save_model()

# Function to predict sentiment
def predict_sentiment(text):
    # Load the trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    # Encode and prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Make prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to label
    sentiment = torch.argmax(logits, dim=1).item()
    sentiment_labels = ["negative", "neutral", "positive"]
    return sentiment_labels[sentiment]

# Ask the user for a news headline
user_input = input("Enter a news headline for sentiment analysis: ")

# Predict sentiment
predicted_sentiment = predict_sentiment(user_input)
print(f"The sentiment of the news headline is: {predicted_sentiment}")
