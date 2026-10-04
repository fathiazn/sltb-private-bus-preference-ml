import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="SLTB vs Private Bus Preference",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 SLTB vs Private Bus Preference Prediction")

st.write(
    "Enter passenger information to predict whether the passenger "
    "is more likely to prefer an SLTB or Private bus."
)

# ============================================================
# Load trained model and encoders
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("models/xgboost.joblib")
    label_encoders = joblib.load("models/label_encoders.joblib")
    target_encoder = joblib.load("models/target_encoder.joblib")

    return model, label_encoders, target_encoder


model, label_encoders, target_encoder = load_model()

# ============================================================
# Input form
# ============================================================

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        label_encoders["Gender"].classes_.tolist()
    )

    age_group = st.selectbox(
        "Age Group",
        label_encoders["Age_Group"].classes_.tolist()
    )

    occupation = st.selectbox(
        "Occupation",
        label_encoders["Occupation"].classes_.tolist()
    )

    income = st.selectbox(
        "Monthly Income",
        label_encoders["Monthly_Income_LKR"].classes_.tolist()
    )

    bus_usage = st.selectbox(
        "Bus Usage Frequency",
        label_encoders["Bus_Usage_Frequency"].classes_.tolist()
    )

    bus_type = st.selectbox(
        "Bus Type Used",
        label_encoders["Bus_Type_Used"].classes_.tolist()
    )

    trip_distance = st.selectbox(
        "Trip Distance",
        label_encoders["Trip_Distance"].classes_.tolist()
    )

    travel_purpose = st.selectbox(
        "Travel Purpose",
        label_encoders["Travel_Purpose"].classes_.tolist()
    )

    wait_time = st.selectbox(
        "Wait Time",
        label_encoders["Wait_Time"].classes_.tolist()
    )


with col2:

    service_freq = st.number_input(
        "Service Frequency",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    comfort = st.number_input(
        "Comfort",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    crowding = st.number_input(
        "Crowding",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    cleanliness = st.number_input(
        "Cleanliness",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    safety = st.number_input(
        "Safety",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    ac = st.number_input(
        "AC",
        min_value=0.0,
        value=1.0,
        step=1.0
    )

    driver = st.number_input(
        "Driver",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    reliability = st.number_input(
        "Reliability",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

# ============================================================
# Prediction
# ============================================================

if st.button("🔮 Predict Bus Preference", use_container_width=True):

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age_Group": age_group,
        "Occupation": occupation,
        "Monthly_Income_LKR": income,
        "Bus_Usage_Frequency": bus_usage,
        "Bus_Type_Used": bus_type,
        "Trip_Distance": trip_distance,
        "Travel_Purpose": travel_purpose,
        "Wait_Time": wait_time,
        "Service_freq": service_freq,
        "Fare": fare,
        "Comfort": comfort,
        "Crowding": crowding,
        "Cleanliness": cleanliness,
        "Safety": safety,
        "AC": ac,
        "Driver": driver,
        "Reliability": reliability
    }])

    # Encode categorical columns
    for column, encoder in label_encoders.items():

        if column in input_data.columns:
            input_data[column] = encoder.transform(
                input_data[column]
            )

    # Make prediction
    prediction = model.predict(input_data)[0]

    prediction_label = target_encoder.inverse_transform(
        [prediction]
    )[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    probability_df = pd.DataFrame({
        "Bus Type": target_encoder.classes_,
        "Probability": probabilities
    })

    # ========================================================
    # Display result
    # ========================================================

    st.subheader("Prediction Result")

    if prediction_label == "SLTB":
        st.success(
            f"🚌 Predicted Preference: **{prediction_label}**"
        )
    else:
        st.info(
            f"🚌 Predicted Preference: **{prediction_label}**"
        )

    st.write("Prediction probabilities:")

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    probability_df["Probability"] = (
        probability_df["Probability"].astype(str) + "%"
    )

    st.table(probability_df)