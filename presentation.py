import streamlit as st

def presentation_page():
    st.title("Презентация проекта")
    st.header("Слайд 1: Введение")
    st.write("Проект предиктивного обслуживания оборудования.")
    st.header("Слайд 2: Датасет")
    st.write("Используется датасет AI4I 2020 с 10 000 записей.")
    st.header("Слайд 3: Модели")
    st.write("Logistic Regression, Random Forest, XGBoost, SVM.")
    st.header("Слайд 4: Результаты")
    st.write("Оценка по точности, ROC-AUC и др.")