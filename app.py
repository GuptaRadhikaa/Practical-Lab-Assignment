# Streamlit App for Smart City Policy Prediction
# Created for: Radhika Gupta
# --------------------------------------------------
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

st.set_page_config(layout="wide", page_title="Smart City Policy & Safety Dashboard")

st.title("Smart City Policy, Safety & Environmental Intelligence Dashboard")
st.markdown("AI-driven Smart Policy Classification using IoT, Public Safety & Environmental Indicators — **Radhika Gupta**")

# ---------------------------
# Load Dataset
# ---------------------------
# using the local path from your environment
DATA_PATH = "smart_city_n_dataset.csv"

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        try:
            return pd.read_excel(path)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None

df = load_data(DATA_PATH)

if df is None:
    st.stop()

# ---------------------------
# FULL EDA SECTION (added)
# ---------------------------
st.header("Exploratory Data Analysis (EDA)")

st.subheader("1. Columns & Shape")
st.write("Columns:", list(df.columns))
st.write("Shape:", df.shape)

st.subheader("2. df.info()")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("3. df.describe()")
st.dataframe(df.describe())

st.subheader("4. Missing Values")
st.dataframe(df.isnull().sum())

st.subheader("5. Random Samples")
st.dataframe(df.sample(5))

st.subheader("6. Histograms of All Numeric Columns")
# create histograms and show correct figure in streamlit
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()

fig = plt.gcf()   # ✔️ get the actual figure that df.hist() created
st.pyplot(fig)    # ✔️ display the real figure

plt.close(fig)    # cleanup


# ---------------------------
# Dataset Overview (collapsible)
# ---------------------------
with st.expander("Dataset Overview (head & quick info)", expanded=False):
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.write("Columns:", list(df.columns))
    st.write("Missing values:", df.isnull().sum())

# ---------------------------
# Drop NA + One-hot encoding
# ---------------------------
# (we perform modeling on a cleaned copy)
df_proc = df.dropna().copy()
categorical_cols = df_proc.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
# ensure target and numeric columns are preserved
if 'Smart_Policy_Status' in categorical_cols:
    try:
        df_proc['Smart_Policy_Status'] = df_proc['Smart_Policy_Status'].astype(int)
        categorical_cols.remove('Smart_Policy_Status')
    except Exception:
        pass
if categorical_cols:
    df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True)

st.write("After preprocessing columns (preview):", list(df_proc.columns)[:30])

# ---------------------------
# Define target + features
# ---------------------------
target_col = "Smart_Policy_Status"
if target_col not in df_proc.columns:
    st.error("Target column missing after preprocessing.")
    st.stop()

y = df_proc[target_col]
X = df_proc.drop(columns=[target_col], errors='ignore')

# keep only numeric features
X = X.select_dtypes(include=[np.number])

# ---------------------------
# Train/Test Split controls in sidebar
# ---------------------------
st.sidebar.header("Modeling options")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)
n_estimators = st.sidebar.slider("Trees (RandomForest)", 50, 300, 100)

# show feature count
st.sidebar.write("Features (numeric):", X.shape[1])

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100.0, random_state=42, stratify=y if len(np.unique(y))>1 else None
)

# ---------------------------
# Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------
st.header("Model Evaluation")
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("Feature Importances")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_names = X.columns[indices]

fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.bar(feat_names, importances[indices])
ax2.set_xticklabels(feat_names, rotation=90)
ax2.set_ylabel("Importance")
ax2.set_xlabel("Feature")
st.pyplot(fig2)
plt.close(fig2)

# Optional: permutation importance (slower)
if st.checkbox("Compute permutation importance (slower)"):
    with st.spinner("Running permutation importance..."):
        perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0, n_jobs=-1)
        perm_df = pd.DataFrame({'feature': X.columns, 'importance_mean': perm.importances_mean}).sort_values('importance_mean', ascending=False)
        st.dataframe(perm_df.head(30))

# ---------------------------
# Download Predictions
# ---------------------------
results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred

csv = results.to_csv(index=False)
st.download_button("Download Predictions CSV", csv, "predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("EDA displayed above uses the raw loaded dataset. Modeling is performed on a cleaned copy (rows with NA dropped and categorical columns one-hot encoded). If you want different imputation or preprocessing, tell me and I will update the pipeline.")
