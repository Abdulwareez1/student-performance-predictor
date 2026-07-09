import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from src.utils import load_data, preprocess_data, save_model

def train_and_compare():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(df)
    
    print("Training optimized RandomForest with hyperparameter tuning...")
    
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid, cv=5, n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Best Accuracy: {acc:.4f}")
    print("Best Parameters:", grid_search.best_params_)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Feature Importance
    plt.figure(figsize=(12, 8))
    feat_importance = pd.Series(
        best_model.feature_importances_, 
        index=df.drop('GradeClass', axis=1).columns
    ).nlargest(10)
    feat_importance.plot(kind='barh')
    plt.title('Top 10 Most Important Features for Grade Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✅ Feature importance plot saved as 'feature_importance.png'")
    
    save_model(best_model, scaler, encoders)
    return best_model

if __name__ == "__main__":
    train_and_compare()