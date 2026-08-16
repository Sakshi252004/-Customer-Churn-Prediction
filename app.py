import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.high-risk {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #ff4b4b;
    background-color: #ffe5e5;
}

.medium-risk {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #f5b041;
    background-color: #fff4d6;
}

.low-risk {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #2ecc71;
    background-color: #e5f7ed;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Customer Retention Dashboard</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Customer Churn Excel",
    type=["xlsx"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Dashboard Features")

st.sidebar.write("📂 Upload Excel")
st.sidebar.write("📋 View Customer Data")
st.sidebar.write("📈 Customer Analytics")
st.sidebar.write("👤 Select Customer")
st.sidebar.write("✏️ Enter Customer Details")
st.sidebar.write("🔮 Predict Churn")
st.sidebar.write("📊 Churn Probability")
st.sidebar.write("🚦 Risk Classification")
st.sidebar.write("💡 Retention Recommendation")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "CustomerID",
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
    "Contract",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "Churn"
]


# ============================================================
# BEFORE FILE UPLOAD
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload customer_churn.xlsx using the sidebar."
    )

    st.header("🚀 Customer Churn Prediction System")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 📂 Upload Data

            Upload your customer churn Excel file.
            """
        )

    with col2:

        st.info(
            """
            ### 🤖 Machine Learning

            Random Forest predicts customer churn.
            """
        )

    with col3:

        st.info(
            """
            ### 🚦 Risk Analysis

            Customers are classified as:

            Low / Medium / High Risk
            """
        )

    st.stop()


# ============================================================
# READ EXCEL FILE
# ============================================================

try:

    df = pd.read_excel(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Error reading Excel file: {e}"
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "❌ Your Excel file is missing required columns."
    )

    st.write("Missing columns:")

    for column in missing_columns:
        st.write("•", column)

    st.write("")

    st.write("Columns found in your Excel:")

    st.write(list(df.columns))

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

df = df.drop_duplicates()

# Clean text columns

for column in df.select_dtypes(
    include="object"
).columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )


# Clean numeric columns

numeric_columns = [
    "SeniorCitizen",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(0)


# ============================================================
# DASHBOARD KPIs
# ============================================================

total_customers = len(df)

churned_customers = (
    df["Churn"]
    .astype(str)
    .str.lower()
    .eq("yes")
    .sum()
)

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0
    else 0
)

average_monthly_charges = (
    df["MonthlyCharges"].mean()
)

average_tenure = (
    df["Tenure"].mean()
)


# ============================================================
# KPI SECTION
# ============================================================

st.header("📈 Customer Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "👥 Total Customers",
        total_customers
    )

with col2:

    st.metric(
        "🚨 Churned Customers",
        churned_customers
    )

with col3:

    st.metric(
        "📊 Churn Rate",
        f"{churn_rate:.1f}%"
    )

with col4:

    st.metric(
        "💰 Avg Monthly Charges",
        f"{average_monthly_charges:.2f}"
    )


# ============================================================
# CUSTOMER DATA
# ============================================================

st.divider()

st.header("📋 Customer Data")

st.dataframe(
    df,
    use_container_width=True,
    height=350
)


# ============================================================
# DOWNLOAD CLEAN DATA
# ============================================================

st.subheader("⬇️ Download Clean Data")

csv_data = df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Clean CSV",
    data=csv_data,
    file_name="customer_churn_clean.csv",
    mime="text/csv"
)


# ============================================================
# CHURN ANALYSIS
# ============================================================

st.divider()

st.header("📊 Churn Analysis")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Churn Distribution")

    churn_distribution = (
        df["Churn"]
        .value_counts()
    )

    st.bar_chart(
        churn_distribution
    )


with col2:

    st.subheader("Contract vs Churn")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(
        contract_churn
    )


# ============================================================
# CUSTOMER SELECTION
# ============================================================

st.divider()

st.header("👤 Select Customer")

customer_ids = (
    df["CustomerID"]
    .astype(str)
    .tolist()
)

selected_customer_id = st.selectbox(
    "Select Customer ID",
    customer_ids
)

selected_customer = df[
    df["CustomerID"].astype(str)
    == selected_customer_id
].iloc[0]


# ============================================================
# SELECTED CUSTOMER DETAILS
# ============================================================

st.subheader("🔎 Customer Details")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.write("**Customer ID**")
    st.write(
        selected_customer["CustomerID"]
    )

    st.write("**Gender**")
    st.write(
        selected_customer["Gender"]
    )

    st.write("**Senior Citizen**")
    st.write(
        selected_customer["SeniorCitizen"]
    )


with col2:

    st.write("**Partner**")
    st.write(
        selected_customer["Partner"]
    )

    st.write("**Dependents**")
    st.write(
        selected_customer["Dependents"]
    )

    st.write("**Tenure**")
    st.write(
        f"{selected_customer['Tenure']} months"
    )


with col3:

    st.write("**Contract**")
    st.write(
        selected_customer["Contract"]
    )

    st.write("**Internet Service**")
    st.write(
        selected_customer["InternetService"]
    )

    st.write("**Payment Method**")
    st.write(
        selected_customer["PaymentMethod"]
    )


with col4:

    st.write("**Monthly Charges**")
    st.write(
        f"{selected_customer['MonthlyCharges']:.2f}"
    )

    st.write("**Total Charges**")
    st.write(
        f"{selected_customer['TotalCharges']:.2f}"
    )

    st.write("**Tech Support**")
    st.write(
        selected_customer["TechSupport"]
    )


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

def train_model(data):

    model_data = data.copy()

    # Remove customer ID

    model_data = model_data.drop(
        "CustomerID",
        axis=1
    )

    # Convert target

    model_data["Churn"] = (
        model_data["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    # Remove invalid target rows

    model_data = model_data.dropna(
        subset=["Churn"]
    )

    X = model_data.drop(
        "Churn",
        axis=1
    )

    y = model_data["Churn"]

    # Convert categorical data

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # Need both classes

    if y.nunique() < 2:

        raise ValueError(
            "Excel must contain both Churn = Yes "
            "and Churn = No."
        )

    # Split data

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # Random Forest

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    # Accuracy

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (
        model,
        X.columns,
        accuracy
    )


# ============================================================
# SELECTED CUSTOMER PREDICTION
# ============================================================

st.divider()

st.header("🔮 Predict Selected Customer")

predict_button = st.button(
    "🔮 PREDICT CHURN",
    use_container_width=True
)

if predict_button:

    try:

        with st.spinner(
            "Training Random Forest model..."
        ):

            model, model_columns, accuracy = (
                train_model(df)
            )

            # Prepare customer

            customer_input = pd.DataFrame(
                [selected_customer.to_dict()]
            )

            # Remove ID and target

            customer_input = customer_input.drop(
                ["CustomerID", "Churn"],
                axis=1
            )

            # One-hot encoding

            customer_input = pd.get_dummies(
                customer_input,
                drop_first=True
            )

            # Match training columns

            customer_input = customer_input.reindex(
                columns=model_columns,
                fill_value=0
            )

            # Prediction

            prediction = model.predict(
                customer_input
            )[0]

            probability = model.predict_proba(
                customer_input
            )[0][1]

        probability_percent = (
            probability * 100
        )


        # ====================================================
        # RISK CLASSIFICATION
        # ====================================================

        if probability >= 0.70:

            risk = "HIGH RISK"
            risk_class = "high-risk"

        elif probability >= 0.40:

            risk = "MEDIUM RISK"
            risk_class = "medium-risk"

        else:

            risk = "LOW RISK"
            risk_class = "low-risk"


        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            "✅ Prediction completed!"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction == 1:

                st.error(
                    "🚨 LIKELY TO CHURN"
                )

            else:

                st.success(
                    "✅ LIKELY TO STAY"
                )


        with col2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )


        with col3:

            st.markdown(
                f"""
                <div class="{risk_class}">
                    <h2>{risk}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.subheader(
            "📊 Churn Probability"
        )

        st.progress(
            float(probability)
        )


        # ====================================================
        # MODEL ACCURACY
        # ====================================================

        st.subheader(
            "🤖 Model Performance"
        )

        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.subheader(
            "💡 Recommended Action"
        )

        if probability >= 0.70:

            st.error(
                "🚨 HIGH RISK: "
                "Contact the customer immediately. "
                "Consider retention offers, discounts, "
                "personalized support, or contract incentives."
            )

        elif probability >= 0.40:

            st.warning(
                "⚠️ MEDIUM RISK: "
                "Monitor this customer closely and "
                "provide proactive customer support."
            )

        else:

            st.success(
                "✅ LOW RISK: "
                "Customer is currently unlikely to churn. "
                "Continue normal engagement."
            )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# ============================================================
# MANUAL CUSTOMER PREDICTION
# ============================================================

st.divider()

st.header("✏️ Enter New Customer Details")

st.write(
    "Enter customer information below to predict "
    "the churn risk of a new customer."
)


with st.form("new_customer_form"):

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        new_gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        new_senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        new_partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        new_dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        new_tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

        new_contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        new_internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        new_payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        new_monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        new_total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

        new_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

        new_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )


    submit_prediction = st.form_submit_button(
        "🔮 PREDICT NEW CUSTOMER",
        use_container_width=True
    )


# ============================================================
# NEW CUSTOMER PREDICTION
# ============================================================

if submit_prediction:

    try:

        with st.spinner(
            "Training model..."
        ):

            model, model_columns, accuracy = (
                train_model(df)
            )

            new_customer = pd.DataFrame([{

                "Gender": new_gender,

                "SeniorCitizen": new_senior,

                "Partner": new_partner,

                "Dependents": new_dependents,

                "Tenure": new_tenure,

                "Contract": new_contract,

                "InternetService": new_internet,

                "MonthlyCharges": new_monthly,

                "TotalCharges": new_total,

                "PaymentMethod": new_payment,

                "TechSupport": new_support,

                "OnlineSecurity": new_security

            }])

            # Encode

            new_customer = pd.get_dummies(
                new_customer,
                drop_first=True
            )

            # Match training columns

            new_customer = new_customer.reindex(
                columns=model_columns,
                fill_value=0
            )

            # Prediction

            prediction = model.predict(
                new_customer
            )[0]

            probability = model.predict_proba(
                new_customer
            )[0][1]


        probability_percent = (
            probability * 100
        )


        # ====================================================
        # RISK
        # ====================================================

        if probability >= 0.70:

            risk = "HIGH RISK"
            risk_class = "high-risk"

        elif probability >= 0.40:

            risk = "MEDIUM RISK"
            risk_class = "medium-risk"

        else:

            risk = "LOW RISK"
            risk_class = "low-risk"


        # ====================================================
        # RESULT
        # ====================================================

        st.success(
            "✅ New customer prediction completed!"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            if prediction == 1:

                st.error(
                    "🚨 LIKELY TO CHURN"
                )

            else:

                st.success(
                    "✅ LIKELY TO STAY"
                )


        with col2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )


        with col3:

            st.markdown(
                f"""
                <div class="{risk_class}">
                    <h2>{risk}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Probability

        st.subheader(
            "📊 Churn Probability"
        )

        st.progress(
            float(probability)
        )


        # Recommendation

        st.subheader(
            "💡 Recommended Action"
        )

        if probability >= 0.70:

            st.error(
                "🚨 HIGH RISK: "
                "Immediate retention action is recommended."
            )

        elif probability >= 0.40:

            st.warning(
                "⚠️ MEDIUM RISK: "
                "Monitor and engage the customer proactively."
            )

        else:

            st.success(
                "✅ LOW RISK: "
                "Continue normal customer engagement."
            )


    except Exception as e:

        st.error(
            f"❌ New customer prediction error: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Python + Machine Learning + Streamlit"
)