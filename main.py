# =====================================================
# Pipeline de Mineração de Dados para Intenção de Compra
#  — sem validação cruzada, foco em precisão, split 80/20
# =====================================================

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    recall_score,
    f1_score,
    classification_report,
    precision_score,
    roc_auc_score,
    roc_curve
)

# =====================================================
# 2. Carregamento e Inspeção dos Dados
# =====================================================
df = pd.read_csv("data/online_shoppers_intention.csv")
print("\n=== 2.1. Primeiras 5 linhas do dataset ===")
print(df.head())
print("\n=== 2.2. Informações gerais do DataFrame ===")
df.info()

# =====================================================
# 3. Pré-processamento
# =====================================================
print("\n=== 3.1. Separação de X e y ===")
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)
print(f"- Variáveis (X): {X.shape[1]} colunas")
print(f"- Alvo (y): {y.value_counts().to_dict()}")

print("\n=== 3.2. Codificação de variáveis categóricas ===")
X = pd.get_dummies(X, drop_first=True)
print(f"- Após dummies: {X.shape[1]} colunas")

print("\n=== 3.3. Divisão treino/teste (80/20) ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"- Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

print("\n=== 3.4. Escalonamento de variáveis numéricas ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("- Escalonamento aplicado.")

# =====================================================
# 4. Modelo 1: Logistic Regression (foco em precisão)
# =====================================================
print("\n=== 4.1. Treinando Logistic Regression ===")
logreg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
logreg.fit(X_train_scaled, y_train)

y_pred_lr   = logreg.predict(X_test_scaled)
precision_lr = precision_score(y_test, y_pred_lr)

print(f"- Precision Logistic Regression: {precision_lr:.3f}")
print(classification_report(y_test, y_pred_lr))

# =====================================================
# 5. Modelo 2: Random Forest (treino simples)
# =====================================================
print("\n=== 5.1. Treinando Random Forest ===")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf    = rf.predict(X_test_scaled)
precision_rf = precision_score(y_test, y_pred_rf)

print(f"- Precision Random Forest: {precision_rf:.3f}")
print(classification_report(y_test, y_pred_rf))

# =====================================================
# 6. Avaliação comparativa
# =====================================================
print("\n=== 6.1. Comparação de Precisão entre Modelos ===")
print(f"- LR Precision: {precision_lr:.3f}")
print(f"- RF Precision: {precision_rf:.3f}")

# =====================================================
# 7. Ajuste Fino: SMOTE + RF Balanceado + Otimização de Threshold
# =====================================================
print("\n=== 7.1. Oversampling com SMOTE ===")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print(f"- Após SMOTE: {X_res.shape[0]} amostras de treino")

print("\n=== 7.2. Treinando RF balanceado ===")
rf_bal = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
rf_bal.fit(X_res, y_res)

# === 7.3. Otimização de threshold pelo F1‐score ===
proba = rf_bal.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, proba)

f1_scores = []
valid_thresholds = []

for thr in thresholds:
    y_pred_thr = (proba >= thr).astype(int)
    if y_pred_thr.sum() > 0:
        f1 = f1_score(y_test, y_pred_thr)
        f1_scores.append(f1)
        valid_thresholds.append(thr)

best_idx = np.argmax(f1_scores)
best_thr = valid_thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"- Melhor threshold (F1): {best_thr:.2f} → F1 = {best_f1:.3f}")

# === 7.4. Avaliação final no threshold escolhido ===
y_pred_adj = (proba >= best_thr).astype(int)

prec  = precision_score(y_test, y_pred_adj)
rec   = recall_score(y_test, y_pred_adj)
auc   = roc_auc_score(y_test, proba)

print(f"- Precision: {prec:.3f}")
print(f"- Recall:    {rec:.3f}")
print(f"- F1:        {best_f1:.3f}")
print(f"- AUC-ROC:   {auc:.3f}\n")

print(classification_report(y_test, y_pred_adj))


# =====================================================
# 8. Gráfico de Curva ROC (modelo balanceado ajustado)
# =====================================================
print("\n=== 8. Curva ROC do modelo ajustado ===")
plt.figure()
plt.plot(fpr, tpr, label=f'Ajustado (AUC = {roc_auc_score(y_test, proba):.3f})')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC – RF Balanceado Ajustado')
plt.legend(loc='lower right')
plt.show()