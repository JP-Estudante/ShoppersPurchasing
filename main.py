# 1. Imports e configuração inicial
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 2. Leitura do CSV (coloque o arquivo em data/online_shoppers_intention.csv)
df = pd.read_csv("data/online_shoppers_intention.csv")

# 3. Pré-processamento básico
#    - separa alvo
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

#    - converte categóricas com get_dummies
X = pd.get_dummies(X, drop_first=True)

#    - divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

#    - padroniza variáveis numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. Modelo baseline: Regressão Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Avaliação
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC: {auc:.3f}")
