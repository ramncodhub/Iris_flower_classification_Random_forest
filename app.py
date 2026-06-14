import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Random Forest Dashboard",
    layout="wide"
)

st.title("🌸 Iris Flower Classification Dashboard")

# --------------------------------------------------
# CACHE DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    y = iris.target

    return iris, X, y

iris, X, y = load_data()

df = X.copy()
df["species"] = [iris.target_names[i] for i in y]

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "model": None,
    "accuracy": None,
    "cm": None,
    "report": None,
    "cv_scores": None,
    "importance_df": None,
    "train_size": None,
    "test_size": None,
    "flower": None,
    "probabilities": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.header("1️⃣ Dataset Preview")

rows = st.slider(
    "Rows to Display",
    min_value=5,
    max_value=len(df),
    value=10
)

st.dataframe(
    df.head(rows),
    use_container_width=True
)

# --------------------------------------------------
# DATASET STATS
# --------------------------------------------------

st.header("2️⃣ Dataset Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# --------------------------------------------------
# FEATURE DISTRIBUTION
# --------------------------------------------------

st.header("3️⃣ Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    X.columns
)

fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(
    X[feature],
    bins=15
)

ax.set_title(feature)
ax.set_xlabel(feature)
ax.set_ylabel("Frequency")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# HYPERPARAMETERS
# --------------------------------------------------

st.header("4️⃣ Hyperparameter Tuning")

col1, col2, col3 = st.columns(3)

with col1:
    n_estimators = st.slider(
        "Number of Trees",
        10,
        500,
        100
    )

with col2:
    max_depth = st.slider(
        "Max Depth",
        1,
        20,
        5
    )

with col3:
    test_size = st.slider(
        "Test Size",
        0.1,
        0.5,
        0.2
    )

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

if st.button("🚀 Train Model"):

    with st.spinner("Training model..."):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        report = classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=5
        )

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        st.session_state.model = model
        st.session_state.accuracy = accuracy
        st.session_state.cm = cm
        st.session_state.report = report
        st.session_state.cv_scores = cv_scores
        st.session_state.importance_df = importance_df
        st.session_state.train_size = len(X_train)
        st.session_state.test_size = len(X_test)

    st.success("✅ Model trained successfully!")

# --------------------------------------------------
# SHOW RESULTS
# --------------------------------------------------

if st.session_state.model is not None:

    st.header("5️⃣ Train/Test Information")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Training Samples",
            st.session_state.train_size
        )

    with c2:
        st.metric(
            "Testing Samples",
            st.session_state.test_size
        )

    # Accuracy

    st.header("6️⃣ Accuracy")

    st.success(
        f"Accuracy: {st.session_state.accuracy:.4f}"
    )

    # Confusion Matrix

    st.header("7️⃣ Confusion Matrix")

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=st.session_state.cm,
        display_labels=iris.target_names
    )

    disp.plot(ax=ax_cm)

    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Classification Report

    st.header("8️⃣ Classification Report")

    st.text(
        st.session_state.report
    )

    # Cross Validation

    st.header("9️⃣ Cross Validation Scores")

    cv_df = pd.DataFrame({
        "Fold": np.arange(
            1,
            len(st.session_state.cv_scores) + 1
        ),
        "Score": st.session_state.cv_scores
    })

    st.dataframe(
        cv_df,
        use_container_width=True
    )

    st.info(
        f"Average CV Score: {st.session_state.cv_scores.mean():.4f}"
    )

    # Feature Importance

    st.header("🔟 Feature Importance")

    st.dataframe(
        st.session_state.importance_df,
        use_container_width=True
    )

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))

    ax_imp.bar(
        st.session_state.importance_df["Feature"],
        st.session_state.importance_df["Importance"]
    )

    plt.xticks(rotation=45)

    st.pyplot(fig_imp)
    plt.close(fig_imp)

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.header("1️⃣1️⃣ Predict New Flower")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length",
            value=5.1,
            step=0.1
        )

        sepal_width = st.number_input(
            "Sepal Width",
            value=3.5,
            step=0.1
        )

    with col2:
        petal_length = st.number_input(
            "Petal Length",
            value=1.4,
            step=0.1
        )

        petal_width = st.number_input(
            "Petal Width",
            value=0.2,
            step=0.1
        )

    predict_btn = st.form_submit_button(
        "🌼 Predict Flower"
    )

# --------------------------------------------------
# MAKE PREDICTION
# --------------------------------------------------

if predict_btn:

    if st.session_state.model is None:

        st.error(
            "⚠️ Please train the model first."
        )

    else:

        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=X.columns
        )

        prediction = st.session_state.model.predict(
            input_data
        )

        probabilities = st.session_state.model.predict_proba(
            input_data
        )[0]

        st.session_state.flower = iris.target_names[
            prediction[0]
        ]

        st.session_state.probabilities = probabilities

# --------------------------------------------------
# SHOW PREDICTION
# --------------------------------------------------

if st.session_state.flower is not None:

    flower = st.session_state.flower

    flower_icons = {
        "setosa": "🌷",
        "versicolor": "🌺",
        "virginica": "🌹"
    }

    st.success(
        f"{flower_icons.get(flower,'🌸')} Predicted Flower: {flower.upper()}"
    )

    prob_df = pd.DataFrame({
        "Species": iris.target_names,
        "Probability": np.round(
            st.session_state.probabilities,
            4
        )
    })

    st.subheader("Prediction Probabilities")

    st.dataframe(
        prob_df,
        use_container_width=True
    )

    st.bar_chart(
        prob_df.set_index("Species")
    )