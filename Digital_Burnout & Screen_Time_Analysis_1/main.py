# ============================================================
# STEP 1: DATA COLLECTION / IMPORT
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Folder Setup ────────────────────────────────────────────
os.makedirs("outputs",      exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# ── Helper: output path ──────────────────────────────────────
def out(filename):
    return os.path.join("outputs", filename)

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_csv("data/mobile_usage_behavioral_analysis.csv")

# Basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 2a. Check for Missing Values ---
print("Missing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# --- 2b. Engineer Target Variable: Burnout Risk Score ---
df['Burnout_Score'] = (
    df['Daily_Screen_Time_Hours']  * 0.35 +
    df['Social_Media_Usage_Hours'] * 0.30 +
    df['Gaming_App_Usage_Hours']   * 0.20 +
    df['Total_App_Usage_Hours']    * 0.15
)

df['Burnout_Score_Normalized'] = (
    (df['Burnout_Score'] - df['Burnout_Score'].min()) /
    (df['Burnout_Score'].max() - df['Burnout_Score'].min())
) * 100

def classify_burnout(score):
    if score < 25:   return 0   # Low
    elif score < 50: return 1   # Moderate
    elif score < 75: return 2   # High
    else:            return 3   # Extreme

df['Burnout_Risk']  = df['Burnout_Score_Normalized'].apply(classify_burnout)
df['Burnout_Label'] = df['Burnout_Risk'].map({0:'Low', 1:'Moderate', 2:'High', 3:'Extreme'})

print("\nTarget Variable Distribution:")
print(df['Burnout_Label'].value_counts())

# --- 2c. Encode Categorical Variables ---
le_gender   = LabelEncoder()
le_location = LabelEncoder()

df['Gender_Enc']   = le_gender.fit_transform(df['Gender'])
df['Location_Enc'] = le_location.fit_transform(df['Location'])

print("\nGender Encoding:",   dict(zip(le_gender.classes_,   le_gender.transform(le_gender.classes_))))
print("Location Encoding:", dict(zip(le_location.classes_, le_location.transform(le_location.classes_))))

# --- 2d. Select Features ---
feature_cols = [
    'Age',
    'Total_App_Usage_Hours',
    'Daily_Screen_Time_Hours',
    'Number_of_Apps_Used',
    'Social_Media_Usage_Hours',
    'Productivity_App_Usage_Hours',
    'Gaming_App_Usage_Hours',
    'Gender_Enc',
    'Location_Enc'
]

X = df[feature_cols]
y = df['Burnout_Risk']

# --- 2e. Feature Scaling ---
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print("\nFeatures shape:", X.shape)
print("Target shape:  ", y.shape)
print("\nScaled Features (first 3 rows):")
print(X_scaled.head(3).round(3))
print("\nPreprocessing Complete ✓")


# ============================================================
# STEP 3: EDA — SEPARATE PLOTS, ALL SAVED TO outputs/
# ============================================================

from matplotlib.patches import Patch

colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
order  = ['Low', 'Moderate', 'High', 'Extreme']

# --- Plot 1: Burnout Risk Distribution ---
fig, ax = plt.subplots(figsize=(7, 6))
burnout_counts = df['Burnout_Label'].value_counts().reindex(order)
ax.pie(
    burnout_counts,
    labels=burnout_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax.set_title("Burnout Risk Distribution", fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(out("burnout_risk_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/burnout_risk_distribution.png ✓")

# --- Plot 2: Screen Time by Burnout Risk ---
fig, ax = plt.subplots(figsize=(8, 6))
data_by_risk = [df[df['Burnout_Label'] == l]['Daily_Screen_Time_Hours'].values for l in order]
bp = ax.boxplot(data_by_risk, labels=order, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Screen Time by Burnout Risk", fontsize=15, fontweight='bold')
ax.set_ylabel("Hours / Day", fontsize=12)
ax.set_xlabel("Burnout Risk Level", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out("screen_time_by_burnout_risk.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/screen_time_by_burnout_risk.png ✓")

# --- Plot 3: Age Distribution ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df['Age'], bins=20, color='#3498db', edgecolor='white', alpha=0.85)
ax.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean Age: {df['Age'].mean():.1f}")
ax.set_title("Age Distribution of Users", fontsize=15, fontweight='bold')
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out("age_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/age_distribution.png ✓")

# --- Plot 4: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(11, 9))
corr_cols = feature_cols + ['Burnout_Risk']
corr = df[corr_cols].corr()
sns.heatmap(
    corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f',
    annot_kws={'size': 8}, linewidths=0.5, square=True
)
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig(out("feature_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/feature_correlation_heatmap.png ✓")

# --- Plot 5: Social Media vs Gaming Scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
risk_colors = df['Burnout_Risk'].map({0:'#2ecc71', 1:'#f39c12', 2:'#e74c3c', 3:'#8e44ad'})
ax.scatter(
    df['Social_Media_Usage_Hours'], df['Gaming_App_Usage_Hours'],
    c=risk_colors, alpha=0.6, s=30, edgecolors='none'
)
ax.set_title("Social Media vs Gaming Usage by Risk", fontsize=15, fontweight='bold')
ax.set_xlabel("Social Media Usage (hrs)", fontsize=12)
ax.set_ylabel("Gaming Usage (hrs)", fontsize=12)
legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, order)]
ax.legend(handles=legend_elements, title="Risk Level", fontsize=10)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(out("social_media_vs_gaming_scatter.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/social_media_vs_gaming_scatter.png ✓")

# --- Plot 6: Burnout Risk by Gender ---
fig, ax = plt.subplots(figsize=(8, 6))
gender_burnout = df.groupby(['Gender', 'Burnout_Label']).size().unstack(fill_value=0)
gender_burnout = gender_burnout[order]
x     = np.arange(len(gender_burnout.index))
width = 0.2
for i, (col, color) in enumerate(zip(gender_burnout.columns, colors)):
    ax.bar(x + i * width, gender_burnout[col], width,
           label=col, color=color, alpha=0.85, edgecolor='white')
ax.set_title("Burnout Risk by Gender", fontsize=15, fontweight='bold')
ax.set_xlabel("Gender", fontsize=12)
ax.set_ylabel("Number of Users", fontsize=12)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(gender_burnout.index, fontsize=12)
ax.legend(title="Risk Level", fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out("burnout_risk_by_gender.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/burnout_risk_by_gender.png ✓")

print("\n✅ All 6 EDA plots saved to outputs/")


# ============================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 45)
print("  TRAIN-TEST SPLIT SUMMARY")
print("=" * 45)
print(f"\nTotal Samples   : {len(X_scaled)}")
print(f"Training Samples: {len(X_train)} (80%)")
print(f"Testing Samples : {len(X_test)}  (20%)")

print("\n--- Class Distribution in Training Set ---")
for idx, count in y_train.value_counts().sort_index().items():
    label = {0:'Low', 1:'Moderate', 2:'High', 3:'Extreme'}[idx]
    print(f"  {label:<10}: {count} samples ({count/len(y_train)*100:.1f}%)")

print("\n--- Class Distribution in Test Set ---")
for idx, count in y_test.value_counts().sort_index().items():
    label = {0:'Low', 1:'Moderate', 2:'High', 3:'Extreme'}[idx]
    print(f"  {label:<10}: {count} samples ({count/len(y_test)*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Train-Test Split Overview", fontsize=15, fontweight='bold')

colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
labels = ['Low', 'Moderate', 'High', 'Extreme']

axes[0].pie(
    [len(X_train), len(X_test)],
    labels=['Train (80%)', 'Test (20%)'],
    autopct='%1.1f%%',
    colors=['#3498db', '#e74c3c'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title("Overall Split", fontsize=12, fontweight='bold')

train_counts = [y_train[y_train == i].count() for i in range(4)]
axes[1].bar(labels, train_counts, color=colors, edgecolor='white', alpha=0.85)
axes[1].set_title("Class Distribution — Train", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Risk Level")
axes[1].set_ylabel("Count")
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(train_counts):
    axes[1].text(i, v + 3, str(v), ha='center', fontsize=10, fontweight='bold')

test_counts = [y_test[y_test == i].count() for i in range(4)]
axes[2].bar(labels, test_counts, color=colors, edgecolor='white', alpha=0.85)
axes[2].set_title("Class Distribution — Test", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Risk Level")
axes[2].set_ylabel("Count")
axes[2].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(test_counts):
    axes[2].text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(out("train_test_split.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/train_test_split.png ✓")
print("\nTrain-Test Split Complete ✓")


# ============================================================
# STEP 5: MODEL TRAINING
# ============================================================

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ── Random Forest ────────────────────────────────────────────
print("=" * 45)
print("  TRAINING MODEL 1: RANDOM FOREST")
print("=" * 45)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("  Random Forest Trained ✓")
print(f"  Trees built   : {rf_model.n_estimators}")
print(f"  Features used : {rf_model.n_features_in_}")
print(f"  Classes       : {list(rf_model.classes_)}")

# ── XGBoost ──────────────────────────────────────────────────
print("\n" + "=" * 45)
print("  TRAINING MODEL 2: XGBOOST")
print("=" * 45)

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("  XGBoost Trained ✓")
print(f"  Boosting rounds : {xgb_model.n_estimators}")
print(f"  Features used   : {xgb_model.n_features_in_}")
print(f"  Classes         : {list(xgb_model.classes_)}")

# ── Feature Importance Plot ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance — Both Models", fontsize=15, fontweight='bold')

feature_names = [
    'Age', 'Total App Usage', 'Daily Screen Time',
    'Apps Used', 'Social Media', 'Productivity',
    'Gaming', 'Gender', 'Location'
]

rf_importance = rf_model.feature_importances_
rf_indices    = np.argsort(rf_importance)[::-1]
axes[0].barh(
    [feature_names[i] for i in rf_indices][::-1],
    rf_importance[rf_indices][::-1],
    color='#00d4aa', edgecolor='white', alpha=0.85
)
axes[0].set_title("Random Forest", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Importance Score")
axes[0].grid(axis='x', linestyle='--', alpha=0.5)
for i, v in enumerate(rf_importance[rf_indices][::-1]):
    axes[0].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

xgb_importance = xgb_model.feature_importances_
xgb_indices    = np.argsort(xgb_importance)[::-1]
xgb_sorted_features = [feature_names[i] for i in xgb_indices]
xgb_sorted_values   = xgb_importance[xgb_indices]
axes[1].barh(
    xgb_sorted_features[::-1],
    xgb_sorted_values[::-1],
    color='#7c5cfc', edgecolor='white', alpha=0.85
)
axes[1].set_title("XGBoost", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Importance Score")
axes[1].grid(axis='x', linestyle='--', alpha=0.5)
for i, v in enumerate(xgb_sorted_values[::-1]):
    axes[1].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(out("feature_importance.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/feature_importance.png ✓")

print("\n✅ Both Models Trained Successfully!")
print("\n--- Top 3 Features (XGBoost) ---")
for i in range(3):
    print(f"  {i+1}. {xgb_sorted_features[i]:<20}: {xgb_sorted_values[i]:.4f}")


# ============================================================
# STEP 6: MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score
)

rf_preds  = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
rf_proba  = rf_model.predict_proba(X_test)
xgb_proba = xgb_model.predict_proba(X_test)

rf_acc  = accuracy_score(y_test, rf_preds)
xgb_acc = accuracy_score(y_test, xgb_preds)
rf_auc  = roc_auc_score(y_test, rf_proba,  multi_class='ovr', average='macro')
xgb_auc = roc_auc_score(y_test, xgb_proba, multi_class='ovr', average='macro')

print("=" * 45)
print("  STEP 6: MODEL EVALUATION")
print("=" * 45)
print(f"\n  {'Model':<20} {'Accuracy':>10}")
print(f"  {'-'*32}")
print(f"  {'Random Forest':<20} {rf_acc*100:>9.2f}%")
print(f"  {'XGBoost':<20} {xgb_acc*100:>9.2f}%")
print(f"\n  {'Model':<20} {'ROC-AUC (macro)':>16}")
print(f"  {'-'*38}")
print(f"  {'Random Forest':<20} {rf_auc:>16.4f}")
print(f"  {'XGBoost':<20} {xgb_auc:>16.4f}")

print("\n" + "=" * 45)
print("  CLASSIFICATION REPORT — RANDOM FOREST")
print("=" * 45)
print(classification_report(y_test, rf_preds))

print("=" * 45)
print("  CLASSIFICATION REPORT — XGBOOST")
print("=" * 45)
print(classification_report(y_test, xgb_preds))

# ── Confusion Matrices ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Confusion Matrices — Both Models", fontsize=15, fontweight='bold')

for ax, preds, title in zip(
    axes,
    [rf_preds, xgb_preds],
    ["Random Forest", "XGBoost"]
):
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

plt.tight_layout()
plt.savefig(out("confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/confusion_matrices.png ✓")

# ── Accuracy & AUC Bar Comparison ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Model Comparison — Accuracy & ROC-AUC", fontsize=14, fontweight='bold')

model_names = ["Random Forest", "XGBoost"]
bar_colors  = ["#00d4aa", "#7c5cfc"]

bars = axes[0].bar(model_names, [rf_acc, xgb_acc], color=bar_colors, edgecolor='white', width=0.4)
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Accuracy", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Score")
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, [rf_acc, xgb_acc]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val*100:.2f}%", ha='center', fontsize=11, fontweight='bold')

bars = axes[1].bar(model_names, [rf_auc, xgb_auc], color=bar_colors, edgecolor='white', width=0.4)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("ROC-AUC (Macro)", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Score")
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, [rf_auc, xgb_auc]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.4f}", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(out("model_comparison.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/model_comparison.png ✓")

print("\n" + "=" * 45)
print("  EVALUATION SUMMARY")
print("=" * 45)
print(f"  {'Metric':<25} {'RF':>8} {'XGB':>8}")
print(f"  {'-'*43}")
print(f"  {'Accuracy':<25} {rf_acc*100:>7.2f}% {xgb_acc*100:>7.2f}%")
print(f"  {'ROC-AUC (macro)':<25} {rf_auc:>8.4f} {xgb_auc:>8.4f}")
best = "Random Forest" if rf_acc >= xgb_acc else "XGBoost"
print(f"\n  🏆 Best Model (by Accuracy): {best}")
print("\n✅ Step 6 Complete — Evaluation Done!")


# ============================================================
# STEP 7: SAVE MODELS
# ============================================================

import joblib

joblib.dump(xgb_model, "saved_models/xgb_model.pkl")
joblib.dump(rf_model,  "saved_models/rf_model.pkl")
joblib.dump(scaler,    "saved_models/scaler.pkl")
print("\n✅ Models saved to saved_models/")

print("\n" + "=" * 45)
print("  PROJECT STRUCTURE")
print("=" * 45)
print("""
  Digital_Burnout_Analysis/
  ├── data/
  │   └── mobile_usage_behavioral_analysis.csv
  ├── outputs/
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
  ├── saved_models/
  │   ├── xgb_model.pkl
  │   ├── rf_model.pkl
  │   └── scaler.pkl
  └── main.py
""")