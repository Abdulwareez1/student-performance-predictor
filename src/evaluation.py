import joblib
from src.utils import load_data, preprocess_data

def evaluate_model(model_path='models/best_model.joblib'):
    data = joblib.load(model_path)
    model = data['model']
    
    df = load_data()
    _, X_test, _, y_test, _, _ = preprocess_data(df)
    
    y_pred = model.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))