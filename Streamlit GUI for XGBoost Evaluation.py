import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gzip
import io
import random

# Set page config
st.set_page_config(page_title="XGBoost Model Evaluation", layout="centered")

st.title("🔍 XGBoost Model Evaluation On Network Anomaly")
st.write("Evaluate your saved XGBoost model using a test dataset (KDD Cup 1999 format).")

# Upload model
st.sidebar.header("📦 Load Saved Model")
model_file = st.sidebar.file_uploader("Upload `XGBoost Model`", type=["pkl"])

# Upload test data
st.sidebar.header("🧪 Load Test Dataset")
test_file = st.sidebar.file_uploader("Upload test dataset", type=["gz", "csv", "xlsx", "xls", "txt"])

# Manual input section
st.sidebar.header("📝 Manual Input for Prediction")
manual_input = st.sidebar.checkbox("Enable Manual Input")

if manual_input:
    # Randomize button logic
    if "randomize" not in st.session_state:
        st.session_state.randomize = False

    if st.sidebar.button("🎲 Randomize Inputs"):
        st.session_state.randomize = True
    else:
        st.session_state.randomize = st.session_state.get("randomize", False)

    # Randomized or default values
    protocol_type_value = random.choice(["tcp", "udp", "icmp"]) if st.session_state.randomize else "tcp"
    service_value = random.choice(["http", "smtp", "ftp", "dns", "other"]) if st.session_state.randomize else "http"
    flag_value = random.choice(["SF", "S0", "REJ", "RSTO", "RSTOS0", "S1", "S2", "S3", "OTH"]) if st.session_state.randomize else "SF"
    duration_value = random.randint(0, 1000) if st.session_state.randomize else 0
    src_bytes_value = random.randint(0, 10000) if st.session_state.randomize else 0
    dst_bytes_value = random.randint(0, 10000) if st.session_state.randomize else 0

    # Create input fields with possibly randomized values
    protocol_type = st.sidebar.selectbox("Protocol Type", ["tcp", "udp", "icmp"], index=["tcp", "udp", "icmp"].index(protocol_type_value))
    service = st.sidebar.selectbox("Service", ["http", "smtp", "ftp", "dns", "other"], index=["http", "smtp", "ftp", "dns", "other"].index(service_value))
    flag = st.sidebar.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTOS0", "S1", "S2", "S3", "OTH"], index=["SF", "S0", "REJ", "RSTO", "RSTOS0", "S1", "S2", "S3", "OTH"].index(flag_value))
    duration = st.sidebar.number_input("Duration", min_value=0, value=duration_value)
    src_bytes = st.sidebar.number_input("Source Bytes", min_value=0, value=src_bytes_value)
    dst_bytes = st.sidebar.number_input("Destination Bytes", min_value=0, value=dst_bytes_value)

    # Reset randomize flag after using it once
    st.session_state.randomize = False

    # Standard values for other features
    standard_values = {
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 0,
        "srv_count": 0,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 0.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 0,
        "dst_host_srv_count": 0,
        "dst_host_same_srv_rate": 0.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }

if model_file and test_file:
    # Load the model
    XGBModel = pickle.load(model_file)

    # Load test dataset
    if test_file.name.endswith(".gz"):
        with gzip.open(test_file, mode='rt') as gz_file:
            test_data = pd.read_csv(gz_file, header=None)
    else:
        test_data = pd.read_csv(test_file, header=None)

    # Define column names
    kddcup_column_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
    ]
    test_data.columns = kddcup_column_names

    st.success("Model and test data loaded successfully.")

    # Encode categorical columns
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col])

    # Convert label to binary
    test_data['label'] = test_data['label'].apply(lambda x: 1 if x == 'normal.' else 0)

    # Split into features and target
    features = test_data.drop(columns=['label'])
    target = test_data['label']

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create DMatrix
    dtest = xgb.DMatrix(features_scaled)

    # Predict
    pred_prob = XGBModel.predict(dtest)
    pred = (pred_prob > 0.5).astype(int)

    # Show evaluation metrics
    accuracy = accuracy_score(target, pred)
    auc = roc_auc_score(target, pred_prob)
    report = classification_report(target, pred, target_names=['Normal', 'Anomaly'], output_dict=True)
    conf_matrix = confusion_matrix(target, pred)

    st.header("✅ Evaluation Metrics")
    st.metric("Accuracy", f"{accuracy:.7f}")
    st.metric("AUC Score", f"{auc:.7f}")

    st.subheader("📋 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose().round(7))

    st.subheader("🔢 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(target, pred_prob)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], 'r--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True)
    st.pyplot(fig_roc)

    # Manual prediction
    if manual_input:
        # Prepare manual input data with standard values
        manual_data = pd.DataFrame({
            "duration": [duration],
            "protocol_type": [protocol_type],
            "service": [service],
            "flag": [flag],
            "src_bytes": [src_bytes],
            "dst_bytes": [dst_bytes],
            **standard_values  # Add standard values for other features
        })

        # Encode and scale manual input
        for col in ['protocol_type', 'service', 'flag']:
            manual_data[col] = le.fit_transform(manual_data[col])

        manual_data_scaled = scaler.transform(manual_data)

        # Create DMatrix for manual input
        dmanual = xgb.DMatrix(manual_data_scaled)

        # Predict
        manual_pred_prob = XGBModel.predict(dmanual)
        manual_pred = (manual_pred_prob > 0.5).astype(int)

        st.subheader("🔮 Manual Prediction Result")
        st.write(f"Predicted Probability: {manual_pred_prob[0]:.7f}")
        st.write(f"Predicted Class: {'Anomaly' if manual_pred[0] == 1 else 'Normal'}")

else:
    st.warning("Please upload both the model and a test dataset.")