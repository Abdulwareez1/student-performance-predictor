import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path='data/Student performance data.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {df.shape[0]} students, {df.shape[1]} features")
    return df

def preprocess_data(df):
    target = 'GradeClass'
    X = df.drop([target], axis=1)
    y = df[target]
    
    # Encode categorical
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, encoders

def save_model(model, scaler, encoders, filename='models/best_model.joblib'):
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'encoders': encoders}, filename)
    print(f"✅ Model saved: {filename}")