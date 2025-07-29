import pandas as pd
import matplotlib.pyplot as plt
import lime.lime_tabular
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregamento do dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
colunas = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings",
    "EmploymentSince", "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
    "ResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
    "NumberCredits", "Job", "LiablePeople", "Telephone", "ForeignWorker", "Target"
]
df = pd.read_csv(url, sep=' ', names=colunas)

# 2. Ajuste da coluna alvo, 1 = bom pagador, 0 = mau pagador
df['Target'] = df['Target'].map({1: 1, 2: 0})

# 3. Separar colunas categóricas e numéricas
X = df.drop(columns=['Target'])
y = df['Target']
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 4. Pré-processamento com ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 5. Pipeline com modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 6. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Treinamento
pipeline.fit(X_train, y_train)

# 10. Avaliar
y_pred = pipeline.predict(X_test)
print("---Relatório de Classificação---\n", (classification_report(y_test, y_pred)))
print("---Matrix de Confusão---\n", (confusion_matrix(y_test, y_pred)))

# 11. LIME - Criar explainer com os dados transformados
# Precisamos transformar os dados manualmente para LIME
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['classifier']

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# 10. Criar o LIME explainer com os dados transformados
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_transformed,
    feature_names=feature_names,
    class_names=["Mau pagador", "Bom pagador"],
    mode='classification'
)

# 11. Escolher uma instância para explicar
i = 10
exp = explainer.explain_instance(
    X_test_transformed[i],
    model.predict_proba,  
    num_features=10
)

# 12. Salvar explicações em imagens para analises posterioras
importances = model.feature_importances_
features = feature_names
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)
feat_df['grupo'] = 'importância' 

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, y='Feature', x='Importance',
            hue='grupo', palette='colorblind', dodge=False)
plt.title('Importância Global de Features rf')
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap='Blues')
plt.title("Matriz de Confusão rf")
plt.tight_layout()
plt.show()

fig = exp.as_pyplot_figure()
fig.savefig('explicacoes cliente 10.png', bbox_inches='tight')
plt.show()