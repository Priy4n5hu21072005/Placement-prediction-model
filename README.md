# 🎓 Placement Chance Predictor (Machine Learning Project)

This project predicts the **probability of a student getting placed** based on academic and skill-related features using **Supervised Machine Learning**.

---

## 📌 Project Overview

Campus placements depend on multiple factors like CGPA, internships, skills, projects, etc.
This ML model analyzes those features and predicts whether a student is **likely to be placed (1)** or **not placed (0)**.

✔ Beginner-friendly
✔ Real-world use case
✔ Perfect for ML mini / major project

---

## 🧠 ML Concepts Used

* Data Cleaning & Preprocessing
* Feature Encoding
* Train-Test Split
* Supervised Learning (Classification)
* Model Evaluation

---

## 📁 Project Structure

```
Placement_Chance_Predictor/
│
├── dataset/
│   └── Campus_Selection.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.py
│   └── 04_Evaluation.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

* **Source:** Campus Placement Dataset
* **Target Column:** `Placement`

  * `1` → Placed
  * `0` → Not Placed

Features may include:

* CGPA
* Internships
* Skills
* Projects
* Certifications
* Aptitude Scores

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <https://github.com/Priy4n5hu21072005/Placement-prediction-model.git>
cd Placement_Chance_Predictor
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Preprocessing

```bash
jupyter notebook 02_Preprocessing.ipynb
```

### 4️⃣ Train the Model

```bash
python 03_Model_Training.py
```

### 5️⃣ Evaluate the Model

```bash
python 04_Evaluation.py
```

---

## 📈 Model Performance

```
Accuracy : 95.34%

Precision, Recall, F1-score:
- Class 0 (Not Placed): 0.92
- Class 1 (Placed): 0.97
```

✔ High accuracy
✔ Balanced classification

---

## 🚀 Future Improvements

* Add **placement probability (%)**
* Deploy using **Streamlit / Flask**
* Try advanced models (XGBoost, Random Forest)
* Add resume-based features

---

## 🧑💻 Author
**Priyanshu**
AI & ML Enthusiast
