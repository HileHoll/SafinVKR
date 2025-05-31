import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def analysis_and_model_page():
    st.title("Анализ данных и обучение моделей")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Датасет загружен:", df.head())

        # Предобработка
        df = df.drop(['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])
        X = df.drop('Target', axis=1)
        y = df['Target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Обучение моделей
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC(probability=True)
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            }

        # Визуализация
        st.subheader("Результаты моделей")
        for name, result in results.items():
            st.write(f"**{name}**")
            st.write(f"Точность: {result['accuracy']:.2f}")
            st.write("Отчет классификации:")
            st.text(result['classification_report'])
            fig, ax = plt.subplots()
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # Предсказание для новых данных
        st.subheader("Предсказание для новых данных")
        input_data = st.text_input("Введите данные (через запятую):")
        if input_data:
            input_array = scaler.transform([list(map(float, input_data.split(',')))])
            prediction = models["Random Forest"].predict(input_array)
            st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Без отказа'}")