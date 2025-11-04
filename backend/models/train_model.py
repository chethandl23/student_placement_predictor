import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.exceptions import ConvergenceWarning
import joblib
import os
import warnings
import pandas as pd 

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# FIX: Use the exact absolute path for saving/loading
MODEL_DIR = r'C:\code crafters\backend\models' 
PROCESSED_DATA_PATH = os.path.join(MODEL_DIR, 'processed_data.csv')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'model.pkl')

def train_and_compare_models():
    """
    Loads processed data, trains and compares multiple classification models, 
    and saves the best one.
    """
    print("--- Starting Model Training and Comparison ---")
    
    # 1. Load Processed Data (Reads from the fixed absolute path)
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}.")
        print("Please ensure data_processor.py ran successfully and created the file at that location.")
        return

    # 2. Separate Features and Target (Unchanged)
    X = df.drop('Placement', axis=1)
    y = df['Placement']

    # 3. Split Data and Define Models (Unchanged)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1

    # 4. Train, Evaluate, and Compare Models (Unchanged)
    print("\n--- Model Evaluation Results ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] 
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        results[name] = {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'F1 Score': f1, 'Model': model}
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC Score: {roc_auc:.4f}")
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name

    # 5. Print Comparison Table (Unchanged)
    print("\n" + "="*50)
    print("           MODEL COMPARISON SUMMARY")
    print("="*50)
    comparison_df = pd.DataFrame({
        'Accuracy': [results[m]['Accuracy'] for m in models],
        'F1 Score': [results[m]['F1 Score'] for m in models],
        'ROC AUC': [results[m]['ROC AUC'] for m in models]
    }, index=models.keys())
    print(comparison_df.sort_values(by='ROC AUC', ascending=False).to_markdown(floatfmt=".4f"))
    print("\n" + "="*50)

    # 6. Save the Best Model (Saves to the fixed absolute path)
    best_model = results[best_model_name]['Model']
    joblib.dump(best_model, MODEL_SAVE_PATH)
    
    print(f"\nSelected Model: {best_model_name} (ROC AUC: {best_roc_auc:.4f})")
    print(f"The final predictor model has been saved to '{MODEL_SAVE_PATH}'")
    print("--- Model Training Complete ---")

if __name__ == '__main__':
    train_and_compare_models()