import torch
from transformers import BertTokenizer, BertForSequenceClassification
import shap
import pandas as pd

# Load dataset
df = pd.read_csv("data/generated_dataset.csv")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Label'].unique()))
model.load_state_dict(torch.load("models/sentence_transformer.pt", map_location=torch.device('cpu')))
model.eval()

# Encode a few sentences for SHAP
sentences = df['Transformed'].tolist()[:10]  # take first 10 for demo
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Function for SHAP
def f(x):
    x = torch.tensor(x, dtype=torch.long)
    with torch.no_grad():
        outputs = model(x)
        return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

# Use SHAP's KernelExplainer
explainer = shap.KernelExplainer(f, inputs['input_ids'].numpy())
shap_values = explainer.shap_values(inputs['input_ids'].numpy(), nsamples=50)

# Display SHAP values
for i, sentence in enumerate(sentences):
    print(f"\nSentence: {sentence}")
    print("SHAP values:", shap_values[0][i])  # show class 0 for simplicity
