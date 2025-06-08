import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/final_fraud_predictions_sample.csv")
    return df

df = load_data()

st.title("Fraud Detection & Anomaly Analytics Engine")

# KPIs
total_tx = len(df)
total_fraud = df['Fraud'].sum()
percent_fraud = (total_fraud / total_tx) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Fraudulent Transactions", f"{total_fraud:,}")
col3.metric("Fraud Rate", f"{percent_fraud:.3f}%")

st.markdown("---")

# Model View
selected_model = st.selectbox("Select Model", ["Isolation Forest", "Autoencoder", "LOF"])

if selected_model == "Isolation Forest":
    pred_col = 'IF_Pred'
elif selected_model == "Autoencoder":
    pred_col = 'Autoencoder_Pred'
else:
    pred_col = 'LOF_Pred'

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(df['Fraud'], df[pred_col])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {selected_model}")
ax.grid(axis='y', linestyle='--', alpha=0.3)

col1, col2, col3 = st.columns([1, 2, 1]) 
with col2:
    st.pyplot(fig)

# Fraud Explorer
st.markdown("### 🔍 High-Risk Transactions")
filtered = df[df[pred_col] == 1]
st.dataframe(filtered.head(100))

st.markdown("---")
st.markdown("Built by Thamizhvaanan Ilango | Streamlit App Demo")