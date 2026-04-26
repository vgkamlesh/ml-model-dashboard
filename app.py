import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


st.set_page_config(page_title="ML Model Comparison Dashboard", layout="wide")

st.title("ML Model Comparison Dashboard")
st.write("Compare multiple machine learning classification models on built-in or uploaded datasets.")


st.sidebar.title("Dataset Selection")

dataset_option = st.sidebar.selectbox(
    "Choose Dataset",
    ["Iris", "Breast Cancer", "Wine", "Upload CSV"]
)


if dataset_option == "Iris":
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

elif dataset_option == "Breast Cancer":
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

elif dataset_option == "Wine":
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

else:
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Upload a CSV file to continue.")
        st.stop()


st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])


target_column = st.selectbox("Select Target Column", df.columns)

test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

selected_models = st.multiselect(
    "Select Models",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    default=[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ]
)


if st.button("Train and Compare Models"):
    data = df.copy()
    data = data.dropna()

    label_encoders = {}

    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    if len(y.unique()) < 2:
        st.error("Target column must have at least 2 classes.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = []
    trained_models = {}

    for name in selected_models:
        model = all_models[name]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test
        }

    results_df = pd.DataFrame(results)

    st.subheader("Model Comparison Table")
    st.dataframe(results_df)

    best_model = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

    st.success(
        f"Best Model: {best_model['Model']} with Accuracy {best_model['Accuracy']:.4f}"
    )

    st.subheader("Accuracy Comparison")

    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["Accuracy"])
    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("All Metrics Comparison")

    fig2, ax2 = plt.subplots()
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
        kind="bar",
        ax=ax2
    )
    ax2.set_ylabel("Score")
    ax2.set_title("Model Metrics Comparison")
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    st.subheader("Confusion Matrix")

    selected_cm_model = st.selectbox(
        "Select Model for Confusion Matrix",
        list(trained_models.keys())
    )

    cm = confusion_matrix(
        trained_models[selected_cm_model]["y_test"],
        trained_models[selected_cm_model]["y_pred"]
    )

    fig3, ax3 = plt.subplots()
    ax3.imshow(cm)
    ax3.set_title(f"Confusion Matrix - {selected_cm_model}")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig3)