import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch

# Paths to preprocessed files
ENTITY_PATH = "../data/entities.csv"

def load_data():
    entities = pd.read_csv(ENTITY_PATH)
    return entities

def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples["sentence"],  # Fixed: Use "sentence" as per your CSV column
        padding=True,
        truncation=True,
        max_length=128,
        is_split_into_words=False
    )
    labels = []
    for i, label_set in enumerate(examples["entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label_to_id[label_set[word_id]])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_data():
    entities = load_data()

    # Split the data
    train_data, test_data = train_test_split(entities, test_size=0.2, random_state=42)

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    unique_labels = list(set(label for label_list in entities["entities"] for label in label_list))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    tokenized_train = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id), batched=True)
    tokenized_test = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id), batched=True)

    return {"train": tokenized_train, "test": tokenized_test}, tokenizer, label_to_id

def train_model(tokenized_dataset, tokenizer, label_to_id):
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_to_id))

    training_args = TrainingArguments(
        output_dir="./ner-model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.3,
        logging_dir="./logs",
        logging_steps=10,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    return trainer, model

def evaluate_model(trainer, label_to_id):
    # Get predictions from the model
    predictions, labels, _ = trainer.predict(tokenized_dataset["test"])

    # Convert predictions to label indices
    predicted_labels = predictions.argmax(-1)

    # Initialize lists for evaluation
    true_labels = []
    pred_labels = []

    # Flatten the predictions and labels while filtering out special tokens
    for sentence_labels, sentence_preds, sentence_input_ids in zip(labels, predicted_labels, tokenized_dataset["test"]["input_ids"]):
        word_ids = tokenizer.convert_ids_to_tokens(sentence_input_ids)  # Map input IDs to tokens
        for label_id, pred_id, word in zip(sentence_labels, sentence_preds, word_ids):
            if word not in ["[CLS]", "[SEP]", "[PAD]"]:  # Ignore special tokens
                if label_id != -100:  # Ignore ignored tokens in true labels
                    true_labels.append(label_id)
                    pred_labels.append(pred_id)

    # Map label IDs back to their names
    id_to_label = {v: k for k, v in label_to_id.items()}
    true_label_names = [id_to_label[label] for label in true_labels]
    pred_label_names = [id_to_label[label] for label in pred_labels]

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(classification_report(true_label_names, pred_label_names))

def extract_entities_from_text(text, tokenizer, model, label_to_id):
    # Check the device of the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the correct device

    # Tokenize input and move inputs to the same device as the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input tensors to the same device

    # Get predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Map predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id_to_label = {v: k for k, v in label_to_id.items()}
    entities = []
    for token, pred_id in zip(tokens, predictions[0].tolist()):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:  # Ignore special tokens
            entities.append((token, id_to_label.get(pred_id, "O")))

    return entities

if __name__ == "__main__":
    print("Preparing data...")
    tokenized_dataset, tokenizer, label_to_id = prepare_data()

    print("Training model...")
    trainer, model = train_model(tokenized_dataset, tokenizer, label_to_id)

    print("Evaluating model...")
    evaluate_model(trainer, label_to_id)

    print("Saving model...")
    model.save_pretrained("./ner_model")
    tokenizer.save_pretrained("./ner_model")

    print("Testing on custom input...")
    test_text = "This document is prepared for Acme Corp on January 1, 2023."
    entities = extract_entities_from_text(test_text, tokenizer, model, label_to_id)
    print(f"Entities in text: {entities}")
