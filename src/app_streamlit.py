import streamlit as st
import torch
import shap
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------------------
# Load model and tokenizer
# ------------------------------
MODEL_PATH = "models/sentence_transformer.pt"
MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ------------------------------
# Define labels
# ------------------------------
labels = [
    "Direct to Indirect Speech",
    "Active to Passive",
    "Negative to Positive",
    "Passive to Active",
    "Positive to Negative",
    "Indirect to Direct Speech"
]

# ------------------------------
# Safe SHAP prediction wrapper
# ------------------------------
def model_predict_safe(inputs):
    """
    SHAP sometimes passes masked numeric arrays, so convert everything to list of strings
    """
    # If numpy array, replace with placeholder words
    if isinstance(inputs, np.ndarray):
        inputs = [" ".join(["word"]*len(inputs[0])) for _ in range(len(inputs))]
    # Ensure list of strings
    if isinstance(inputs, str):
        inputs = [inputs]
    inputs = [str(x) for x in inputs]

    # Tokenize and predict
    tokenized = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokenized)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
    return probs

# ------------------------------
# SHAP explainer
# ------------------------------
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(model_predict_safe, masker)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Sentence Transformation Classifier with SHAP Explanations")

sentence = st.text_input("Enter a sentence to classify:")

if sentence:
    # Prediction
    probs = model_predict_safe([sentence])[0]
    pred_idx = np.argmax(probs)
    pred_label = labels[pred_idx]

    st.subheader("Prediction")
    st.write(f"**Transformation Type:** {pred_label}")
    st.write(f"**Confidence Score:** {probs[pred_idx]:.2f}")

    # SHAP explanation
    shap_values = explainer([sentence])
    st.subheader("Word Contribution (SHAP values)")
    shap.plots.text(shap_values[0])  # Original SHAP text visualization

    # ------------------------------
    # Word-level SHAP values table and bar chart
    # ------------------------------
    words = shap_values[0].data
    values = shap_values[0].values[:, pred_idx]  # Fix: use all words for predicted class

    # Create a DataFrame
    shap_df = pd.DataFrame({
        "Word": words,
        "SHAP Value": values
    })

    # Color coding for positive/negative contributions
    def color_shap(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'

    st.subheader("Word-level SHAP Values Table")
    st.dataframe(shap_df.style.applymap(color_shap, subset=["SHAP Value"]))

    st.subheader("SHAP Values Bar Chart")
    st.bar_chart(shap_df.set_index("Word"))
