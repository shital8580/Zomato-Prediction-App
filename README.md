# Zomato-Prediction-App

🍽️ Zomato Prediction App
A Machine Learning-powered web application to predict restaurant ratings, customer preferences, and analyze food trends using Zomato data. 

📌 Project Overview
This project leverages data science & machine learning techniques to help businesses and customers analyze restaurant performance based on historical Zomato data. Users can input various features (such as cuisine type, location, and price range) to get predicted ratings and insights.

🌟 Key Features:
✅ Predicts restaurant ratings based on multiple factors
✅ Interactive data visualization for deeper insights
✅ Streamlit-based UI for easy deployment and access
✅ Uses pre-trained ML models for fast predictions

⚙ Tech Stack
Category	Tools / Technologies Used
Frontend	Streamlit
Backend	Python
Machine Learning	Scikit-Learn, Joblib
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
Deployment	Streamlit Cloud, GitHub

🚀 Installation & Setup Guide
🔹 1️⃣ Clone the Repository
git clone https://github.com/shital8580/Zomato-Prediction-App.git
cd Zomato-Prediction-App

🔹 2️⃣ Create a Virtual Environment (Recommended)
python -m venv env
source env/bin/activate  # Mac/Linux
env\Scripts\activate     # Windows

🔹 3️⃣ Install Required Dependencies
pip install -r requirements.txt

🔹 4️⃣ Run the Application
streamlit run app.py
Once the server starts, open the displayed URL in a browser to access the app.

🌍 Deployment on Streamlit Cloud
To deploy the app on Streamlit Cloud, follow these steps:

1️⃣ Push your code to GitHub:
git add .
git commit -m "Deploying Zomato Prediction App"
git push origin main

2️⃣ Go to Streamlit Cloud
3️⃣ Click ‘New App’ and connect your GitHub repository
4️⃣ Select the branch & app.py as the main script
5️⃣ Deploy and share your live app link! 🎉

📊 Dataset Used
The app is powered by real-world restaurant data from Zomato, which includes:

🍕 Restaurant Name & Location
⭐ Customer Ratings
🍲 Cuisines & Food Type
💰 Price Range
📝 Customer Reviews & Feedback
📈 Machine Learning Model
The Zomato Prediction App uses ML algorithms to analyze restaurant trends and predict ratings based on:

✔ Linear Regression – Predicting numerical values (e.g., ratings)
✔ Random Forest – Handling categorical & numerical inputs
✔ Logistic Regression – Binary classification problems

The trained models are saved using joblib for quick deployment & inference.

💡 Suggestions & feature requests? Open an Issue on GitHub!

🛠️ Troubleshooting & Common Errors
❌ ModuleNotFoundError: No module named 'joblib'
🔹 Solution: Install missing dependencies
pip install -r requirements.txt
❌ Streamlit App Not Deploying on Cloud
🔹 Solution: Ensure your requirements.txt is updated and includes all dependencies.
