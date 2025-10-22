import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/generated_dataset.csv")
labels = df["Label"].unique()
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
df["LabelEnc"] = df["Label"].map(label2id)

# Split dataset (same as train.py)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(df["Transformed"], df["LabelEnc"], stratify=df["LabelEnc"], test_size=0.2, random_state=42)

# Dataset class
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
model.load_state_dict(torch.load("models/sentence_transformer.pt", map_location=torch.device('cpu')))
model.eval()

# DataLoader
test_dataset = SentenceDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {k:v for k,v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch['labels'].tolist())

# Metrics
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=labels))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
