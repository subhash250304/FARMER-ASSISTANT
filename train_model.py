import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

def train_model_and_scaler(
    cleaned_data_path="C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Data\\processed\\crop_data_cleaned.csv", 
    model_output_path="C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Models\\crop_model.pkl",
    scaler_output_path="C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Scripts\\models\scaler.pkl"
):
    """
    Loads cleaned data, scales the features, trains a RandomForestClassifier,
    and crucially saves both the trained model and the scaler object.
    """
    # --- 1. Load Cleaned Data ---
    if not os.path.exists(cleaned_data_path):
        print(f"❌ Error: Cleaned data file not found at '{cleaned_data_path}'")
        print("➡️ Please run the preprocess.py script first.")
        return

    print(f"🔹 Loading cleaned data from '{cleaned_data_path}'...")
    df = pd.read_csv(cleaned_data_path)

    # --- 2. Define Features (X) and Target (y) ---
    if 'label' not in df.columns:
        print("❌ Error: 'label' column not found.")
        return
    X = df.drop("label", axis=1)
    y = df["label"]

    # --- 3. Split Data before Scaling ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 4. Scale Features and Save the Scaler ---
    print("\n🔹 Scaling features using StandardScaler...")
    scaler = StandardScaler()
    # Fit on training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)
        with open(scaler_output_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved successfully to '{scaler_output_path}'")
    except Exception as e:
        print(f"❌ Error saving the scaler: {e}")
        return

    # --- 5. Train the Model on Scaled Data ---
    print("\n🔹 Training RandomForestClassifier model on scaled data...")
    model = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=15)
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- 6. Evaluate Model Performance ---
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Model accuracy on the test set: {accuracy:.4f}")

    # --- 7. Save the Trained Model ---
    try:
        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Model saved successfully to '{model_output_path}'")
    except Exception as e:
        print(f"❌ Error saving the model: {e}")

if __name__ == "__main__":
    train_model_and_scaler()

