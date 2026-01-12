import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    df = pd.read_csv('creditcard.csv')
    print(f"Distribuição original das classes:\n{df['Class'].value_counts(normalize=True)}")

    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Aplicando SMOTE... Aguarde")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Treino antes do SMOTE:{len(y_train)} transações")
    print(f"Treino depois do SMOTE:{len(y_train_resampled)} transações")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    print("Treinando o modelo, essa ação pode demorar um pouco")
    clf.fit(X_train_resampled, y_train_resampled)
    
    y_pred = clf.predict(X_test)


    print("----- Relatório de Classificação -----")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz Confusa (Dados de Testes Reais)')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()