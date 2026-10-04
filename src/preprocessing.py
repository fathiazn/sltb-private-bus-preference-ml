import pandas as pd
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = [
    'Gender',
    'Age_Group',
    'Occupation',
    'Monthly_Income_LKR',
    'Bus_Usage_Frequency',
    'Bus_Type_Used',
    'Trip_Distance',
    'Travel_Purpose',
    'Wait_Time'
]


def load_and_preprocess_data(data_path):
    """Load and preprocess the bus preference dataset."""

    df = pd.read_excel(data_path)

    # Remove matching column if it exists
    if 'match_key' in df.columns:
        df = df.drop(columns=['match_key'])

    # Fill missing Wait_Time values using the mode
    df['Wait_Time'] = df['Wait_Time'].fillna(
        df['Wait_Time'].mode()[0]
    )

    # Keep Age_Group as string
    df['Age_Group'] = df['Age_Group'].astype(str)

    # Encode categorical features
    df_encoded = df.copy()
    label_encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    df_encoded['Bus_Preference'] = target_encoder.fit_transform(
        df_encoded['Bus_Preference']
    )

    # Separate features and target
    X = df_encoded.drop(columns=['Bus_Preference'])
    y = df_encoded['Bus_Preference']

    return X, y, label_encoders, target_encoder


if __name__ == "__main__":
    X, y, _, _ = load_and_preprocess_data(
        "data/bus_preference_data.xlsx"
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("\nTarget values:")
    print(y.value_counts())
