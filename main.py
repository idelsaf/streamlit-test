import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Инициализация модели для классификации изображений (вынесена из условия для оптимизации)
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

# Заголовок приложения
st.title("Распознавание изображений с помощью Hugging Face")

# Добавляем блок с загрузкой изображения от пользователя
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

# Проверка, загружено ли изображение
if uploaded_file is not None:
    try:
        # Открываем изображение с помощью PIL и выводим на экран
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        # Загружаем модель
        classifier = load_classifier()
        
        # Распознавание изображения и запись результатов
        with st.spinner("Распознаю изображение..."):
            results = classifier(image)
        
        st.write("**Результаты распознавания:**")
        for result in results:
            st.write(f"{result['label']}: {result['score']:.2f}")
            
    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {str(e)}")
