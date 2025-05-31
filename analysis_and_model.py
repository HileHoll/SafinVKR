import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def analysis_and_model_page():
    st.title("Анализ данных и обучение моделей")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Датасет загружен:", df.head())
        st.write("Колонки в датасете:", list(df.columns))

        # Проверка наличия столбца 'Target' или аналогичного
        target_column = None
        possible_target_columns = ['Target', 'target', 'Machine failure', 'Failure']
        for col in possible_target_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            st.error("Столбец с целевой переменной ('Target' или аналогичный) не найден. Пожалуйста, проверьте датасет.")
            return

        # Предобработка
        df = df.drop(['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, errors='ignore')
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])
        X = df.drop(target_column, axis=1)
        y = df[target_column]
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
                "classification_report": classification_report(y_test, y_pred)
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
        input_data = st.text_input("Введите данные (через запятую, например: 0,298.1,308.6,1595,39.5,0):")
        if input_data:
            try:
                input_array = scaler.transform([list(map(float, input_data.split(',')))])
                prediction = models["Random Forest"].predict(input_array)
                st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Без отказа'}")
            except Exception as e:
                st.error(f"Ошибка ввода: {e}")