Smart Farm Assistant - Full-Stack Application
This project provides a complete solution for crop and pesticide recommendations, featuring a machine learning backend powered by FastAPI and a modern, responsive frontend.

Project Structure
The project is organized into distinct directories for clarity and scalability.

smart-farm-fullstack/
│
├── backend/
│   ├── data/
│   │   ├── raw/
│   │   │   └── Crop_recommendation.csv   <-- Your raw input data
│   │   ├── processed/
│   │   │   └── (empty until you run preprocess.py)
│   │   └── pesticide_dataset.csv         <-- Your pesticide data
│   │
│   ├── models/
│   │   └── (empty until you run train_model.py)
│   │
│   ├── .env                            <-- Your secret API keys
│   ├── app.py                          <-- Main FastAPI server
│   ├── preprocess.py                   <-- Data cleaning & scaling script
│   ├── train_model.py                  <-- Model training script
│   └── requirements.txt                <-- Python dependencies
│
└── frontend/
    └── index.html                      <-- The user interface

🚀 How to Run This Project
Follow these steps in order. All commands should be run from the smart-farm-fullstack/backend/ directory.

Step 1: Set Up the Backend Environment
First, install all the necessary Python libraries.

# Navigate into the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

Step 2: Preprocess the Raw Data
Run the preprocessing script. This will clean your raw data and save a scaled version in the backend/data/processed/ folder.

python preprocess.py

Step 3: Train the Machine Learning Model
Now, run the training script. This will use the processed data to train a RandomForest model and save it as crop_model.pkl in the backend/models/ folder.

python train_model.py

Step 4: Run the FastAPI Server
Your backend is now ready. Start the server.

uvicorn app:app --reload

The server will be running at http://127.0.0.1:8000.

Step 5: View the Frontend
Navigate to the frontend/ directory and open the index.html file in your web browser. The dashboard will connect to your running FastAPI server automatically.