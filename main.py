import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting ML Pipeline...")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# -----------------------------
# 2. DATA CLEANING
# -----------------------------
cols_to_drop = ['User_ID', 'Caffeine_Intake', 'Reaction_Time',
                'Memory_Test_Score', 'Cognitive_Score', 'AI_Predicted_Score']

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

if 'Unnamed: 13' in df.columns:
    df = df.drop(columns=['Unnamed: 13'])

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in ['Gender', 'Diet_Type', 'Exercise_Frequency', 'Risk']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df.drop_duplicates(inplace=True)

# Convert numeric
numeric_cols = ['Age', 'Sleep_Duration', 'Stress_Level', 'Daily_Screen_Time']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# 3. ENCODING
# -----------------------------
df = pd.get_dummies(df, columns=['Gender', 'Diet_Type', 'Exercise_Frequency'], drop_first=True)

from sklearn.preprocessing import LabelEncoder
le_risk = LabelEncoder()
df['Risk'] = le_risk.fit_transform(df['Risk'])

# -----------------------------
# 4. FEATURE & TARGET
# -----------------------------
X = df.drop('Risk', axis=1)
y = df['Risk']

# -----------------------------
# 5. TRAIN TEST SPLIT
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. FEATURE SCALING
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 7. MODELS
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
knn = KNeighborsClassifier()
svm = SVC(probability=True, class_weight='balanced')

# -----------------------------
# 8. HYPERPARAMETER TUNING
# -----------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 21)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_knn.fit(X_train, y_train)

knn_best = grid_knn.best_estimator_
print("Best K:", grid_knn.best_params_)

# -----------------------------
# 9. TRAIN MODELS
# -----------------------------
lr.fit(X_train, y_train)
knn_best.fit(X_train, y_train)
svm.fit(X_train, y_train)

# -----------------------------
# 10. PREDICTIONS
# -----------------------------
y_pred_lr = lr.predict(X_test)
y_pred_knn = knn_best.predict(X_test)
y_pred_svm = svm.predict(X_test)

# -----------------------------
# 11. EVALUATION
# -----------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(name, y_test, y_pred):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

evaluate("Logistic Regression", y_test, y_pred_lr)
evaluate("KNN (Tuned)", y_test, y_pred_knn)
evaluate("SVM", y_test, y_pred_svm)

# -----------------------------
# 12. MODEL COMPARISON
# -----------------------------
models = ['Logistic Regression', 'KNN', 'SVM']
scores = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_svm)
]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.show()

# -----------------------------
# 13. CONFUSION MATRICES
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models_preds = [
    ("Logistic Regression", y_pred_lr),
    ("KNN", y_pred_knn),
    ("SVM", y_pred_svm)
]

for ax, (name, y_pred) in zip(axes, models_preds):
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png")
plt.show()

# -----------------------------
# 14. FEATURE CORRELATION
# -----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.savefig("correlation_heatmap.png")
plt.show()

# -----------------------------
# 15. FEATURE IMPORTANCE (LR)
# -----------------------------
importance = pd.Series(lr.coef_[0], index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance (Logistic Regression)")
plt.savefig("feature_importance.png")
plt.show()

# -----------------------------
# 16. TEST ON NEW DATA
# -----------------------------
test_df = pd.read_csv("test_data_500.csv")

test_df = pd.get_dummies(test_df)
test_df = test_df.reindex(columns=X.columns, fill_value=0)

X_new = scaler.transform(test_df)

predictions = svm.predict(X_new)

labels = le_risk.classes_
test_df['Predicted_Risk'] = [labels[p] for p in predictions]

test_df.to_csv("output_predictions.csv", index=False)

print("\n Predictions saved to output_predictions.csv")
print("Pipeline completed successfully!")


import pickle

pickle.dump(svm, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_risk, open("label_encoder.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))