import os
import re
import spacy
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import nltk

nltk.download('punkt')

# Load SpaCy English NLP model
nlp = spacy.load("en_core_web_sm")

# Directory where synthetic legal documents are stored
data_dir = "../data/"

def load_documents():
    """Load synthetic legal documents from the data directory."""
    documents = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r") as file:
            documents.append(file.read())
    return documents

def clean_text(text):
    """Clean and preprocess text."""
    # Remove special characters, extra spaces, and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()

def tokenize_sentences(documents):
    """Tokenize documents into sentences."""
    sentences = []
    for doc in documents:
        sentences.extend(sent_tokenize(doc))
    return sentences

def extract_entities(text):
    """Extract named entities using SpaCy."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def preprocess_documents():
    """Full preprocessing pipeline."""
    # Load documents
    docs = load_documents()

    # Clean and tokenize text
    cleaned_docs = [clean_text(doc) for doc in docs]
    sentences = tokenize_sentences(cleaned_docs)

    # Extract named entities for analysis
    entity_data = []

    for sentence in sentences:
        entities = extract_entities(sentence)
        if entities:
            entity_data.append({"sentence": sentence, "entities": entities})  # Fixed: Correctly map sentence to entities

    # Return processed sentences and a DataFrame of entities
    return sentences, pd.DataFrame(entity_data)

def split_data(sentences):
    """Split data into training and testing sets."""
    return train_test_split(sentences, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("Loading and preprocessing documents...")
    sentences, entity_df = preprocess_documents()

    # Save the processed data
    entity_df.to_csv("../data/entities.csv", index=False)
    train, test = split_data(sentences)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Save training and testing sentences
    with open("../data/train.txt", "w") as f:
        f.write("\n".join(train))
    with open("../data/test.txt", "w") as f:
        f.write("\n".join(test))

    print("Preprocessing complete. Data saved to the data directory.")
