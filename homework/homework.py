import gzip
import json
import pickle
import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# Cargar los datos
df_train = pd.read_csv("files/input/train.csv")
df_test = pd.read_csv("files/input/test.csv")

# Paso 1: Limpieza de datos
df_train.rename(columns={"default payment next month": "default"}, inplace=True)
df_test.rename(columns={"default payment next month": "default"}, inplace=True)
df_train.drop(columns=["ID"], inplace=True)
df_test.drop(columns=["ID"], inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
df_train["EDUCATION"] = df_train["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
df_test["EDUCATION"] = df_test["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

# Paso 2: Separar variables predictoras y objetivo
x_train, y_train = df_train.drop(columns=["default"]), df_train["default"]
x_test, y_test = df_test.drop(columns=["default"]), df_test["default"]

# Paso 3: Crear pipeline de preprocesamiento y clasificación
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_features = list(x_train.select_dtypes(include=["int64", "float64"]).columns.difference(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("scaler", StandardScaler(), numeric_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA()),
    ("feature_selection", SelectKBest()),
    ("classifier", LinearSVC(dual=False, max_iter=2000)),
])

# Paso 4: Optimización de hiperparámetros con RandomizedSearchCV
param_grid = {
    "pca__n_components": [5, 10, 15],
    "feature_selection__k": [5, 10, 15, 20],
    "classifier__C": [0.01, 0.1, 1, 10],
}

random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_grid, n_iter=10, scoring="balanced_accuracy", cv=5, n_jobs=-1
)

random_search.fit(x_train, y_train)

# Paso 5: Guardar el modelo
with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(random_search, file)

# Paso 6: Calcular y guardar métricas
def save_metrics(dataset_name, y_true, y_pred):
    metrics = {
        "type": "metrics",  # Se agrega esta línea
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    return metrics

y_train_pred = random_search.predict(x_train)
y_test_pred = random_search.predict(x_test)

metrics = [
    save_metrics("train", y_train, y_train_pred),
    save_metrics("test", y_test, y_test_pred)
]

# Paso 7: Guardar matrices de confusión
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

def format_cm_matrix(dataset_name, cm):
    return {
        "type": "cm_matrix", "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1]) if cm.shape[1] > 1 else 0},
        "true_1": {"predicted_0": int(cm[1, 0]) if cm.shape[0] > 1 else 0, "predicted_1": int(cm[1, 1]) if cm.shape == (2, 2) else 0},
    }

metrics.append(format_cm_matrix("train", cm_train))
metrics.append(format_cm_matrix("test", cm_test))

with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    for metric in metrics:
        file.write(json.dumps(metric) + "\n")
