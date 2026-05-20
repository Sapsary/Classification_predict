import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

df = pd.read_csv(r"C:\Users\OMEN\Desktop\Univer\Programming\NLP_Ai\data.csv")

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# train/val/test split (80/10/10)
train_texts, temp_texts, train_class, temp_class = train_test_split(
    df["content"], df["class"], test_size=0.2, stratify=df["class"], random_state=42
)
val_texts, test_texts, val_class, test_class = train_test_split(
    temp_texts, temp_class, test_size=0.5, stratify=temp_class, random_state=42
)

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch, max_len=128):
    return tokenizer(batch, padding="max_length", truncation=True, max_length=max_len)

train_encodings = tokenize(list(train_texts))
val_encodings = tokenize(list(val_texts))
test_encodings = tokenize(list(test_texts))

class NewsDataset(Dataset):
    def __init__(self, encodings, classes):
        self.encodings = encodings
        self.classes = classes

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.classes.iloc[idx]))
        return item

train_dataset = NewsDataset(train_encodings, train_class)
val_dataset = NewsDataset(val_encodings, val_class)
test_dataset = NewsDataset(test_encodings, test_class)

num_labels = len(df["class"].unique())
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=200,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_class.values

print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 0 - "Science", 1 - "world", 2 - "sports", 3 - "business"