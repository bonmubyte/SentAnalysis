'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Function to train and save the model
def train_and_save_model():
    # Load the dataset
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    
    # Check available splits
    print("Available splits in the dataset:", dataset.keys())

    # (Optionally) Create a test split if it doesn't exist
    # ... code to split the dataset ...

    # Preprocess the data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

#    def preprocess_function(examples):
#        return tokenizer(examples["text"], truncation=True, padding=True)

#    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

    
# Uncomment the line below to train and save the model
train_and_save_model()
'''

#below is only an experiment with a much smaller training dataset.

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

def train_and_save_model():
    # Load and prepare the dataset
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    # Reduce the dataset size for quicker training (e.g., use 10% of the training data)
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))  # Adjust the range as needed

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=1,  # Reduced number of epochs
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_small_train_dataset
    )

    trainer.train()

    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

train_and_save_model()


# Function to predict sentiment
def predict_sentiment(text):
    # Load the trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")

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
