# =====================================================
# Pipeline de Mineração de Dados para Intenção de Compra
# =====================================================

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE

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
print(f"- After dummies: {X.shape[1]} colunas")

print("\n=== 3.3. Divisão treino/teste (80/20) ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"- Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

print("\n=== 3.4. Escalonamento de variáveis numéricas ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("- Escalonamento aplicado.")

# =====================================================
# 4. Modelo 1: Logistic Regression (CV AUC)
# =====================================================
print("\n=== 4.1. Treinando Logistic Regression com validação cruzada ===")
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
scores = cross_val_score(
    logreg, X_train_scaled, y_train,
    cv=5, scoring='roc_auc'
)
print(f"- Logistic Regression CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# =====================================================
# 5. Modelo 2: Random Forest (GridSearchCV)
# =====================================================
print("\n=== 5.1. GridSearchCV para Random Forest ===")
rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(f"- Melhores parâmetros RF: {grid.best_params_}")
print(f"- RF CV AUC: {grid.best_score_:.3f}")

# =====================================================
# 6. Avaliação do Melhor RF no Conjunto de Teste
# =====================================================
print("\n=== 6.1. Avaliação no teste ===")
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred))
print(f"- Test AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# =====================================================
# 7. Ajuste Fino: SMOTE + RF Balanceado + Threshold
# =====================================================
print("\n=== 7.1. Oversampling com SMOTE ===")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print(f"- Após SMOTE: {X_res.shape[0]} amostras de treino")

print("\n=== 7.2. Treinando RF balanceado ===")
rf_bal = RandomForestClassifier(
    n_estimators=grid.best_params_['n_estimators'],
    max_depth=grid.best_params_['max_depth'],
    class_weight='balanced', random_state=42
)
rf_bal.fit(X_res, y_res)

print("\n=== 7.3. Otimização de threshold para F1 ===")
proba = rf_bal.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, proba)
f1_scores = [f1_score(y_test, proba >= thr) for thr in thresholds]
best_idx = np.argmax(f1_scores)
best_thr = thresholds[best_idx]
print(f"- Melhor threshold: {best_thr:.2f} (F1 = {f1_scores[best_idx]:.2f})")

print("\n=== 7.4. Avaliação final com threshold ajustado ===")
y_pred_adj = (proba >= best_thr).astype(int)
print(classification_report(y_test, y_pred_adj))
print(f"- AUC-ROC (mesmo threshold): {roc_auc_score(y_test, proba):.3f}")

# =====================================================
# 8. Gráfico de Curva ROC (para apresentação)
# =====================================================
print("\n=== 8. Curva ROC do modelo ajustado ===")
plt.figure()
plt.plot(fpr, tpr, label=f'Ajustado (AUC = {roc_auc_score(y_test, proba):.3f})')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Random Forest Ajustado')
plt.legend(loc='lower right')
plt.show()

