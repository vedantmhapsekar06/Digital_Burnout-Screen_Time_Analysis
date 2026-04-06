# 📱 Digital Burnout & Screen Time Analysis

A machine learning project that analyzes mobile usage patterns to predict **digital burnout risk** levels. It combines exploratory data analysis, feature engineering, and two classification models — Random Forest and XGBoost — with a Flask web app for real-time predictions.

---

## 🗂️ Project Structure

```
Digital_Burnout-Screen_Time_Analysis/
├── data/
│   └── mobile_usage_behavioral_analysis.csv   # Dataset
├── outputs/                                    # Auto-generated EDA & evaluation plots
│   ├── burnout_risk_distribution.png
│   ├── screen_time_by_burnout_risk.png
│   ├── age_distribution.png
│   ├── feature_correlation_heatmap.png
│   ├── social_media_vs_gaming_scatter.png
│   ├── burnout_risk_by_gender.png
│   ├── train_test_split.png
│   ├── feature_importance.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
├── saved_models/                               # Serialized trained models
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
├── templates/
│   └── index.html                             # Frontend UI for predictions
├── main.py                                    # Full ML pipeline (EDA → training → evaluation)
└── app.py                                     # Flask API server
```

---

## 🔍 Overview

The pipeline covers end-to-end ML workflow:

1. **Data Import** — Load and inspect mobile usage behavioral data
2. **Preprocessing** — Engineer a composite Burnout Score, encode categoricals, scale features
3. **EDA** — Generate 6 visualizations covering risk distribution, screen time patterns, demographics, and correlations
4. **Train/Test Split** — 80/20 stratified split to preserve class balance
5. **Model Training** — Train Random Forest (200 trees) and XGBoost (200 rounds)
6. **Evaluation** — Accuracy, ROC-AUC, classification reports, and confusion matrices
7. **Deployment** — Flask REST API with a browser-based prediction interface

---

## 🎯 Target Variable

A **Burnout Score** is computed as a weighted sum of screen time features:

| Feature | Weight |
|---|---|
| Daily Screen Time (hrs) | 35% |
| Social Media Usage (hrs) | 30% |
| Gaming App Usage (hrs) | 20% |
| Total App Usage (hrs) | 15% |

The normalized score (0–100) is then classified into four risk levels:

| Label | Score Range |
|---|---|
| 🟢 Low | 0 – 24 |
| 🟡 Moderate | 25 – 49 |
| 🔴 High | 50 – 74 |
| 🟣 Extreme | 75 – 100 |

---

## 📊 Dataset

**File:** `data/mobile_usage_behavioral_analysis.csv`

| Column | Description |
|---|---|
| `User_ID` | Unique identifier |
| `Age` | User age |
| `Gender` | Male / Female |
| `Total_App_Usage_Hours` | Daily total hours across all apps |
| `Daily_Screen_Time_Hours` | Total daily screen-on time |
| `Number_of_Apps_Used` | Count of distinct apps used |
| `Social_Media_Usage_Hours` | Hours on social media apps |
| `Productivity_App_Usage_Hours` | Hours on productivity apps |
| `Gaming_App_Usage_Hours` | Hours on gaming apps |
| `Location` | City (New York, Chicago, Houston, Phoenix, Los Angeles) |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Digital_Burnout-Screen_Time_Analysis.git
cd Digital_Burnout-Screen_Time_Analysis

pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask flask-cors joblib
```

---

## 🚀 Usage

### Run the full ML pipeline

```bash
python main.py
```

This trains both models, generates all plots in `outputs/`, and saves the models to `saved_models/`.

### Launch the prediction web app

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### API endpoint

```
POST /predict
Content-Type: application/json
```

**Request body:**

```json
{
  "age": 25,
  "total_app": 4.0,
  "screen_time": 7.0,
  "num_apps": 15,
  "social_media": 2.5,
  "productivity": 2.0,
  "gaming": 2.0,
  "gender_enc": 0,
  "location_enc": 0
}
```

**Response:**

```json
{
  "xgb_label": "Moderate",
  "rf_label": "Moderate",
  "xgb_proba": [0.12, 0.65, 0.18, 0.05],
  "rf_proba":  [0.10, 0.60, 0.22, 0.08]
}
```

---

## 🤖 Models

| Model | Algorithm | Key Hyperparameters |
|---|---|---|
| Random Forest | `RandomForestClassifier` | 200 trees, max_depth=12 |
| XGBoost | `XGBClassifier` | 200 rounds, lr=0.1, max_depth=6 |

Both models are evaluated on **Accuracy** and **macro ROC-AUC**. The better-performing model is highlighted in the evaluation summary printed at the end of `main.py`.

---

## 📈 Output Plots

All plots are saved to the `outputs/` folder automatically when you run `main.py`:

- **Burnout Risk Distribution** — Pie chart of class proportions
- **Screen Time by Burnout Risk** — Box plots per risk level
- **Age Distribution** — Histogram with mean marker
- **Feature Correlation Heatmap** — Seaborn heatmap across all features
- **Social Media vs Gaming Scatter** — Coloured by risk level
- **Burnout Risk by Gender** — Grouped bar chart
- **Train/Test Split** — Class balance across splits
- **Feature Importance** — Side-by-side for both models
- **Confusion Matrices** — Side-by-side for both models
- **Model Comparison** — Accuracy and ROC-AUC bar charts

---

## 🛠️ Tech Stack

- **Data:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **ML:** scikit-learn, XGBoost
- **Serialization:** joblib
- **Web:** Flask, Flask-CORS

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
