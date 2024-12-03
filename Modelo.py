import streamlit as st
import numpy as np
import tensorflow as tf

# CSS personalizado para cambiar el fondo
page_bg = """
<style>
    body {
        background-color: #00FFFF; /* Fondo color cian */
        color: black; /* Color de texto */
    }
    .stButton>button {
        background-color: #007ACC; /* Color de los botones */
        color: white;
    }
</style>
"""

# Incluir el estilo en la aplicación
st.markdown(page_bg, unsafe_allow_html=True)

# Título de la aplicación
st.title("Análisis de Señales ECG: Artefacto vs Arritmia")
st.write("Sube uno o más archivos de texto con las características extraídas de tus señales para clasificarlas.")

# Cargar el modelo exportado en formato .h5
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Modelo.h5") #CAMBIAR ESTO

model = load_model()

# Formulario de datos del paciente
st.markdown("<h3 style='color: #2E86C1;'>Formulario</h3>", unsafe_allow_html=True)
with st.form("Formulario"):
    # Cargar archivo de texto
    uploaded_files = st.file_uploader(
        "Sube tus archivos de características (.txt)", 
        type=["txt"], 
        accept_multiple_files=True
    )

    # Botón de enviar
    submit_button = st.form_submit_button("Enviar")

if submit_button:
    if uploaded_files is not None:
        st.write(f"Se han subido {len(uploaded_files)} archivo(s).")
    
        results = []  # Para guardar los resultados de cada archivo

        for uploaded_file in uploaded_files:
            try:
                # Leer las características desde el archivo
                features = np.loadtxt(uploaded_file, delimiter=",")

                # Asegurarse de que las características tengan la forma esperada por el modelo
                if features.ndim == 1:
                    features = features.reshape(1, -1)  # Añadir dimensión batch si es un solo conjunto de características

                # Realizar predicción con el modelo
                prediction = model.predict(features)

                # Interpretar la predicción
                class_names = ["Arritmia", "Artefacto (Rascado)"]  # Ajusta según tus clases
                predicted_class = class_names[np.argmax(prediction, axis=1)[0]]

                # Mostrar resultados
                st.write("Resultados de la predicción (probabilidades):")
                st.write(prediction)
                st.success(f"Clase predicha para {uploaded_file.name}: {predicted_class}")
            
                # Guardar resultado
                results.append({"Archivo": uploaded_file.name, "Clase Predicha": predicted_class})

            except Exception as e:
                st.error(f"Error al procesar el archivo {uploaded_file.name}: {e}")

        # Mostrar resumen de resultados
        if results:
            st.write("Resumen de predicciones:")
            for result in results:
                st.write(f"- **{result['Archivo']}**: {result['Clase Predicha']}")
    else:
        st.error("Por favor, suba un archivo.")