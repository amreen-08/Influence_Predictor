"""
Streamlit App: OTT/Media Influence Score Predictor
Based on final_improved.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Media Influence Score Predictor",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Media Influence Score Predictor")
st.markdown(
    "This app trains a regression model on survey data about OTT/media "
    "watching habits and predicts a viewer's **Influence Score** "
    "(1–10 scale) based on their responses."
)

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()
    return df


# ---------------------------------------------------------------
# Full training pipeline (cached)
# ---------------------------------------------------------------
@st.cache_resource
def train_pipeline(df):
    """
    Replicates the notebook pipeline:
    - Train/test split
    - Label / OneHot encoding
    - Standard scaling
    - LassoCV feature selection
    - Backward elimination via p-values
    - Final OLS model
    """
    x = df.drop(columns="Influence_Score")
    y = pd.DataFrame(df["Influence_Score"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Encode categorical columns
    encoders = {}
    cat_cols = x.select_dtypes(include="object").columns
    for col in cat_cols:
        if col not in ["Gender", "Fav_Genre"]:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            encoders[col] = le
        else:
            ohe = OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            )
            X_train_enc = ohe.fit_transform(X_train[[col]])
            X_test_enc = ohe.transform(X_test[[col]])
            enc_cols = ohe.get_feature_names_out([col])

            X_train_enc = pd.DataFrame(
                X_train_enc, columns=enc_cols, index=X_train.index
            )
            X_test_enc = pd.DataFrame(
                X_test_enc, columns=enc_cols, index=X_test.index
            )

            X_train = X_train.drop(columns=col)
            X_test = X_test.drop(columns=col)

            X_train = pd.concat([X_train, X_train_enc], axis=1)
            X_test = pd.concat([X_test, X_test_enc], axis=1)
            encoders[col] = ohe

    feature_names = X_train.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # VIF table
    vif = pd.DataFrame(
        {
            "feature": X_train_df.columns,
            "VIF": [
                variance_inflation_factor(X_train_df.values, i)
                for i in range(X_train_df.shape[1])
            ],
        }
    ).sort_values("VIF", ascending=False)

    # LassoCV
    lasso_cv = LassoCV(alphas=None, cv=5, random_state=42)
    lasso_cv.fit(X_train_scaled, Y_train.values.ravel())

    kept_features = [
        f for f, c in zip(feature_names, lasso_cv.coef_) if c != 0
    ]
    dropped_features = [
        f for f, c in zip(feature_names, lasso_cv.coef_) if c == 0
    ]

    X_train_lasso = X_train_df[kept_features].reset_index(drop=True)
    X_test_lasso = X_test_df[kept_features].reset_index(drop=True)
    Y_train_reset = Y_train.reset_index(drop=True)

    # Backward elimination via p-value
    X_train_current = X_train_lasso.copy()
    X_test_current = X_test_lasso.copy()
    final_features = list(X_train_current.columns)
    removed_log = []

    while True:
        X_train_ols = sm.add_constant(X_train_current)
        model = sm.OLS(Y_train_reset, X_train_ols).fit()
        p_values = model.pvalues.drop("const")
        max_p = p_values.max()
        if max_p > 0.05:
            feat = p_values.idxmax()
            removed_log.append((feat, float(max_p)))
            X_train_current = X_train_current.drop(columns=[feat])
            X_test_current = X_test_current.drop(columns=[feat])
            final_features.remove(feat)
        else:
            break

    # Metrics
    X_test_ols = sm.add_constant(X_test_current, has_constant="add")
    X_test_ols = X_test_ols[X_train_ols.columns]

    y_train_pred = model.predict(X_train_ols)
    y_test_pred = model.predict(X_test_ols)

    metrics = {
        "train_r2": r2_score(Y_train_reset, y_train_pred),
        "test_r2": r2_score(Y_test, y_test_pred),
        "train_mse": mean_squared_error(Y_train_reset, y_train_pred),
        "test_mse": mean_squared_error(Y_test, y_test_pred),
    }

    errors = Y_test["Influence_Score"].values - y_test_pred.values

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "kept_features": kept_features,
        "dropped_features": dropped_features,
        "final_features": final_features,
        "removed_log": removed_log,
        "vif": vif,
        "metrics": metrics,
        "best_alpha": lasso_cv.alpha_,
        "errors": errors,
        "y_test": Y_test,
        "y_test_pred": y_test_pred,
    }


# ---------------------------------------------------------------
# Sidebar: upload & navigation
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Setup")

uploaded = st.sidebar.file_uploader(
    "Upload `response.csv`", type=["csv"]
)

if uploaded is None:
    st.info("👈 Upload `response.csv` in the sidebar to get started.")
    st.stop()

df = load_data(uploaded)

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Data Overview",
        "🧠 Model Training",
        "🔮 Predict Influence Score",
    ],
)

# ---------------------------------------------------------------
# Train once
# ---------------------------------------------------------------
results = train_pipeline(df)

# ---------------------------------------------------------------
# Page: Data Overview
# ---------------------------------------------------------------
if page == "📊 Data Overview":
    st.header("📊 Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (after dropna)", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Target", "Influence_Score")

    st.subheader("Sample rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Numeric summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distribution of Influence Score")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["Influence_Score"], bins=10, edgecolor="black")
    ax.set_xlabel("Influence Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Correlation of numeric features")
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax
    )
    st.pyplot(fig)

# ---------------------------------------------------------------
# Page: Model Training
# ---------------------------------------------------------------
elif page == "🧠 Model Training":
    st.header("🧠 Model Training Results")

    st.subheader("1. Variance Inflation Factor (VIF)")
    st.dataframe(results["vif"], use_container_width=True)

    st.subheader("2. Lasso Feature Selection")
    st.write(
        f"**Best alpha (via 5-fold CV):** `{results['best_alpha']:.5f}`"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Kept ({len(results['kept_features'])})**")
        st.write(results["kept_features"])
    with c2:
        st.markdown(
            f"**Dropped ({len(results['dropped_features'])})**"
        )
        st.write(results["dropped_features"])

    st.subheader("3. Backward elimination (p > 0.05)")
    if results["removed_log"]:
        st.dataframe(
            pd.DataFrame(
                results["removed_log"],
                columns=["Removed feature", "p-value"],
            ),
            use_container_width=True,
        )
    else:
        st.write("No features removed.")

    st.markdown("**Final features:**")
    st.write(results["final_features"])

    st.subheader("4. Final OLS Model Summary")
    st.text(str(results["model"].summary()))

    st.subheader("5. Performance")
    m = results["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train R²", f"{m['train_r2']:.4f}")
    c2.metric("Test R²", f"{m['test_r2']:.4f}")
    c3.metric("Train MSE", f"{m['train_mse']:.4f}")
    c4.metric("Test MSE", f"{m['test_mse']:.4f}")

    st.subheader("6. Residual distribution (test set)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results["errors"], bins=10, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Prediction error (actual − predicted)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("7. Actual vs Predicted (test set)")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        results["y_test"]["Influence_Score"],
        results["y_test_pred"],
        alpha=0.7,
    )
    lims = [1, 10]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual Influence Score")
    ax.set_ylabel("Predicted Influence Score")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    st.pyplot(fig)

# ---------------------------------------------------------------
# Page: Predict
# ---------------------------------------------------------------
elif page == "🔮 Predict Influence Score":
    st.header("🔮 Predict Your Influence Score")
    st.markdown(
        "Fill out the form below — these are the features the final "
        "model kept after Lasso + p-value pruning."
    )

    final_features = results["final_features"]
    scaler = results["scaler"]
    model = results["model"]
    feature_names = results["feature_names"]

    # --------------------------------------------------
    # Build a full input row so we can reuse the scaler
    # (the scaler was fit on all 25 features)
    # --------------------------------------------------
    with st.form("prediction_form"):
        st.subheader("Your responses")

        col1, col2 = st.columns(2)

        with col1:
            emotional = st.slider(
                "Emotional connection with characters (1–10)",
                1, 10, 6,
                help="How much do you emotionally connect with characters?",
            )
            mindset = st.slider(
                "Mindset / attitude influenced by content (1–10)",
                1, 10, 5,
            )
            career = st.slider(
                "Career-goal influence (1–10)",
                1, 10, 5,
            )

        with col2:
            follow_trends = st.slider(
                "How much do you follow trends? (1–10)",
                1, 10, 5,
            )
            product_bought = st.radio(
                "Have you bought a product because of content you watched?",
                ["No", "Yes"],
                horizontal=True,
            )

        submitted = st.form_submit_button("Predict Influence Score")

    if submitted:
        # Build a mean-valued row for all training features, then
        # overwrite the ones we collected. Because the scaler was fit
        # on the training mean/std, passing the training means for
        # untouched features produces z-scores of 0 for them — which
        # is exactly the baseline behaviour we want.
        means = pd.Series(
            scaler.mean_, index=feature_names
        )
        row = means.copy()

        # Product_Bought was label-encoded: 'No' -> 0, 'Yes' -> 1
        # (sorted alphabetically by LabelEncoder)
        row["Emotional_Connection_Character"] = emotional
        row["Mindset_Attitude"] = mindset
        row["Career_Goals"] = career
        row["Follow_Trends"] = follow_trends
        row["Product_Bought"] = 1 if product_bought == "Yes" else 0

        # Scale
        scaled = scaler.transform(
            pd.DataFrame([row.values], columns=feature_names)
        )
        scaled_df = pd.DataFrame(scaled, columns=feature_names)

        # Keep only final features + constant
        X_new = scaled_df[final_features]
        X_new = sm.add_constant(X_new, has_constant="add")
        X_new = X_new[model.model.exog_names]

        pred = float(model.predict(X_new).iloc[0])
        pred_clipped = float(np.clip(pred, 1, 10))

        st.success(f"### Predicted Influence Score: **{pred_clipped:.2f} / 10**")
        st.caption(
            f"(raw model output before clipping to 1–10: {pred:.2f})"
        )

        # Interpretation
        if pred_clipped < 4:
            msg = "🟢 Low influence — content doesn't strongly sway you."
        elif pred_clipped < 7:
            msg = "🟡 Moderate influence — content shapes some of your choices."
        else:
            msg = "🔴 High influence — content strongly shapes your mindset & habits."
        st.info(msg)

        st.subheader("Model coefficients (scaled features)")
        coef_df = (
            pd.DataFrame(
                {
                    "Feature": model.params.index,
                    "Coefficient": model.params.values,
                    "p-value": model.pvalues.values,
                }
            )
            .round(4)
        )
        st.dataframe(coef_df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built from `final_improved.ipynb` • "
    "Pipeline: OneHot/Label encode → StandardScaler → LassoCV → "
    "backward p-value elimination → OLS."
)