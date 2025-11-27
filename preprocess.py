import pandas as pd
import os

def clean_crop_data(
    input_path="C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Data\\raw\\Crop_recommendation.xls", 
    output_path="C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Data\\processed\\crop_data_cleaned.csv"
):
    """
    Loads the raw dataset, which is a CSV file misnamed as .xls. It handles 
    missing values and saves the cleaned data to a new, standard CSV file.
    """
    # --- 1. Load Dataset ---
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        return None
        
    print(f"🔹 Loading raw data from '{input_path}'...")
    try:
        # FIX: The file is a CSV despite the .xls extension, so use read_csv.
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading the data file: {e}")
        return None
    print("Shape of raw data:", df.shape)

    # --- 2. Handle Missing Values ---
    if df.isnull().sum().sum() > 0:
        print("\n🔹 Handling missing values by filling with the mean...")
        for col in df.select_dtypes(include='number').columns:
            df[col].fillna(df[col].mean(), inplace=True)
        print("Missing values handled.")
    else:
        print("\n🔹 No missing values found.")
    
    # --- 3. Save Cleaned Data as CSV ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # BEST PRACTICE: Save processed data as a standard CSV file
        df.to_csv(output_path, index=False)
        print(f"\n✅ Cleaning complete! Data saved to '{output_path}'")
        return df
    except Exception as e:
        print(f"❌ Error saving cleaned file: {e}")
        return None

if __name__ == "__main__":
    clean_crop_data()

