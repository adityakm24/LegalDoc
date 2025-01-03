import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


ENTITY_PATH = "../data/entities.csv"
entities_df = pd.read_csv(ENTITY_PATH)

all_entities = []

for entity_list in entities_df["entities"]:
    entity_list = eval(entity_list)
    for _, label in entity_list:
        all_entities.append(label)

label_counts = Counter(all_entities)

print("Entity Label Distrubtion: ")
for label, count in label_counts.items():
    print(f"{label}: {count}")
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values(), color="skyblue")
    plt.xlabel("Entity Labels")
    plt.ylabel("Frequency")
    plt.title("Entity Label Distribution")
    plt.xticks(rotation=45)
    plt.show()