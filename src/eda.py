import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_data

def perform_eda():
    df = load_data()
    print(df.head())
    print(df.describe())
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.show()
    
    # Distribution of final grade
    sns.histplot(df['G3'], kde=True)
    plt.title('Distribution of Final Grade (G3)')
    plt.show()

if __name__ == "__main__":
    perform_eda()