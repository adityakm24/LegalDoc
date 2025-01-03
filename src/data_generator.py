from faker import Faker
import random
import os

# Initialize Faker
faker = Faker()

# Directory for storing synthetic data
output_dir = "../data/"
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic legal data
def generate_legal_docs(num_docs=100):
    legal_types = ["Contract", "Agreement", "Compliance Document"]
    docs = []

    for _ in range(num_docs):
        doc_type = random.choice(legal_types)
        content = f"{doc_type}\n\nThis document is prepared for {faker.company()}.\n" \
                  f"Effective Date: {faker.date_this_decade()}\n\nTerms:\n- {faker.sentence()}\n" \
                  f"- {faker.sentence()}\n- {faker.sentence()}\n\n" \
                  f"Signatures:\n{faker.name()}, {faker.job()}"

        docs.append({"type": doc_type, "content": content})

    return docs

# Save documents to text files
def save_docs_to_files(docs):
    for idx, doc in enumerate(docs):
        with open(os.path.join(output_dir, f"doc_{idx + 1}.txt"), "w") as file:
            file.write(doc["content"])

if __name__ == "__main__":
    documents = generate_legal_docs(50)  # Generate 50 synthetic documents
    save_docs_to_files(documents)
    print(f"Generated {len(documents)} documents in {output_dir}")
