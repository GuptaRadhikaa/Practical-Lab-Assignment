# Running the Project Locally
## Step 1: Clone the repository

 
```bash
git clone https://github.com/GuptaRadhikaa/Practical-Lab-Assignment.git
cd Practical-Lab-Assignment
```

## Step 2: Create a virtual environment

Windows


```bash
python -m venv venv
venv\Scripts\activate
```


Mac/Linux


```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install dependencies

```bash
pip install pipenv
pipenv install 
```

## Step 4: Run the Jupyter notebook

```bash
jupyter notebook
```
``` bash
Open:

LAB5_Radhika_05001192022.ipynb
```

## Inside the notebook you can reproduce:

EDA
Preprocessing
Train/test split
Model training
Model evaluation
Saving the model checkpoint

# Saving & Loading the Trained Model
## Saving the model (already implemented in your notebook):

```bash
import joblib
joblib.dump(model, "models/random_forest_model.joblib")
```

## Loading the model:
``` bash
import joblib
loaded_model = joblib.load("models/random_forest_model.joblib")
loaded_model.predict(X_test[:5])
```

# Running the Streamlit Dashboard

Your dashboard script (uploaded as app.py app) can be executed using:
```bash
streamlit run app.py
```

This automatically opens the dashboard at:

http://localhost:8501/

## Dashboard features:
Full EDA
Preprocessing overview
Adjustable model parameters (sidebar)
Model training & evaluation
Download predictions button

## Live demo
```bash
https://practical-lab-assignment-v9o5pbu8h2sqiyq5ptkp27.streamlit.app/
```
