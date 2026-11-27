
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import av
import json
import threading
import os

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from collections import Counter

# CONFIGURACION DE LA PAGINA

st.set_page_config(
    page_title="Reconocimiento de Objetos",
    page_icon="📷",
    layout="wide"
)

st.title("Reconocimiento de Objetos con Deep Learning")

st.markdown("""
Sistema de detección de objetos mediante una red neuronal
profunda YOLO11 preentrenada.
""")

# CARGAR MODELO

@st.cache_resource
def cargar_modelo():
    return YOLO("yolo11n.pt"), threading.Lock()

modelo, bloqueo = cargar_modelo()

# CLASES EN ESPAÑOL

nombres_es = [
    "Persona", "Bicicleta", "Automovil",
    "Motocicleta", "Avion", "Autobus",
    "Tren", "Camion", "Barco",
    "Semaforo", "Hidrante", "Senal de alto",
    "Parquimetro", "Banco", "Pajaro",
    "Gato", "Perro", "Caballo",
    "Oveja", "Vaca", "Elefante",
    "Oso", "Cebra", "Jirafa",
    "Mochila", "Paraguas", "Bolso",
    "Corbata", "Maleta", "Disco volador",
    "Esquis", "Tabla de nieve", "Pelota",
    "Cometa", "Bate de beisbol",
    "Guante de beisbol", "Patineta",
    "Tabla de surf", "Raqueta de tenis",
    "Botella", "Copa", "Taza",
    "Tenedor", "Cuchillo", "Cuchara",
    "Tazon", "Banana", "Manzana",
    "Sandwich", "Naranja", "Brocoli",
    "Zanahoria", "Hot dog", "Pizza",
    "Dona", "Pastel", "Silla",
    "Sofa", "Planta", "Cama",
    "Mesa", "Inodoro", "Televisor",
    "Computadora portatil", "Mouse",
    "Control remoto", "Teclado",
    "Celular", "Microondas", "Horno",
    "Tostadora", "Lavamanos",
    "Refrigerador", "Libro", "Reloj",
    "Florero", "Tijeras", "Oso de peluche",
    "Secadora de cabello", "Cepillo dental"
]

clases_es = dict(enumerate(nombres_es))

# METRICAS DE EVALUACION

st.subheader("Rendimiento del modelo")

if os.path.exists("metricas.json"):

    with open("metricas.json", "r") as archivo:
        metricas = json.load(archivo)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision",
        f"{metricas['Precision']:.2f}%"
    )

    col2.metric(
        "Recall",
        f"{metricas['Recall']:.2f}%"
    )

    col3.metric(
        "mAP50",
        f"{metricas['mAP50']:.2f}%"
    )

    col4.metric(
        "mAP50-95",
        f"{metricas['mAP50-95']:.2f}%"
    )

    st.caption(
        f"Dataset: {metricas.get('dataset', 'No indicado')} | "
        f"Resolucion evaluada: {metricas.get('imgsz', 'No indicada')}"
    )

    st.caption(
        "mAP es la metrica principal de deteccion de objetos. "
        "No equivale al accuracy tradicional."
    )

else:
    st.warning("No se encontro el archivo metricas.json")

# MENU LATERAL

st.sidebar.title("Configuracion")

st.sidebar.info("Modelo: YOLO11n")

st.sidebar.metric("Categorias disponibles", 80)

confianza = st.sidebar.slider(
    "Umbral de confianza",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)

resolucion = st.sidebar.selectbox(
    "Resolucion de procesamiento",
    [320, 480, 640],
    index=0
)

opcion = st.sidebar.radio(
    "Seleccionar modalidad",
    [
        "Cargar imagen",
        "Camara en tiempo real",
        "Informacion del modelo"
    ]
)

# FUNCION DE DETECCION

def detectar_objetos(imagen, umbral, tamano):

    with bloqueo:
        resultados = modelo.predict(
            imagen,
            conf=umbral,
            imgsz=tamano,
            verbose=False
        )

    imagen_resultado = imagen.copy()
    objetos = []

    for resultado in resultados:

        for caja in resultado.boxes:

            x1, y1, x2, y2 = map(
                int, caja.xyxy[0].tolist()
            )

            clase = int(caja.cls.item())
            probabilidad = float(caja.conf.item())

            nombre = clases_es[clase]

            etiqueta = (
                f"{nombre} {probabilidad * 100:.1f}%"
            )

            cv2.rectangle(
                imagen_resultado,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                imagen_resultado,
                etiqueta,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            objetos.append({
                "Objeto": nombre,
                "Confianza": round(
                    probabilidad * 100, 2
                )
            })

    return imagen_resultado, objetos


# OPCION 1: CARGAR IMAGEN

if opcion == "Cargar imagen":

    st.subheader("Deteccion de objetos en imagenes")

    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:

        bytes_imagen = np.frombuffer(
            archivo.getvalue(),
            dtype=np.uint8
        )

        imagen = cv2.imdecode(
            bytes_imagen,
            cv2.IMREAD_COLOR
        )

        if imagen is None:
            st.error("No se pudo leer la imagen")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                imagen,
                channels="BGR",
                caption="Imagen original"
            )

        if st.button(
            "Detectar objetos",
            type="primary"
        ):

            with st.spinner("Procesando imagen..."):

                resultado, objetos = detectar_objetos(
                    imagen,
                    confianza,
                    resolucion
                )

            with col2:

                st.image(
                    resultado,
                    channels="BGR",
                    caption="Objetos detectados"
                )

            st.metric(
                "Total de objetos detectados",
                len(objetos)
            )

            if objetos:

                df = pd.DataFrame(objetos)

                st.subheader("Resultados")

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.metric(
                    "Confianza promedio",
                    f"{df['Confianza'].mean():.2f}%"
                )

                st.subheader("Confianza por objeto")

                st.bar_chart(
                    df.groupby("Objeto")["Confianza"].mean()
                )

                st.subheader("Cantidad por categoria")

                conteo = Counter(
                    objeto["Objeto"]
                    for objeto in objetos
                )

                st.bar_chart(
                    pd.Series(conteo)
                )

                st.download_button(
                    "Descargar resultados CSV",
                    df.to_csv(index=False),
                    "resultados.csv",
                    "text/csv"
                )

            else:
                st.warning("No se detectaron objetos")


# OPCION 2: CAMARA EN TIEMPO REAL

elif opcion == "Camara en tiempo real":

    st.subheader("Reconocimiento mediante camara")

    st.info(
        "Presiona START y permite el acceso "
        "a la camara de tu computadora."
    )

    def procesar_video(frame):

        imagen = frame.to_ndarray(
            format="bgr24"
        )

        resultado, objetos = detectar_objetos(
            imagen,
            confianza,
            resolucion
        )

        cv2.putText(
            resultado,
            f"Objetos detectados: {len(objetos)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            resultado,
            format="bgr24"
        )

    webrtc_streamer(
        key="reconocimiento-objetos",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=procesar_video,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},
                "height": {"ideal": 360},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        async_processing=True
    )


# OPCION 3: INFORMACION DEL MODELO

elif opcion == "Informacion del modelo":

    st.subheader("Arquitectura del modelo")

    st.write("Modelo: YOLO11n")
    st.write("Framework: PyTorch")
    st.write("Dataset de preentrenamiento: COCO")
    st.write("Tipo: Red neuronal profunda")
    st.write("Categorias: 80")

    st.subheader("Clases disponibles")

    df_clases = pd.DataFrame({
        "ID": list(clases_es.keys()),
        "Objeto": list(clases_es.values())
    })

    st.dataframe(
        df_clases,
        use_container_width=True
    )

    with st.expander("Capas de la red neuronal"):

        st.code(str(modelo.model))


# PIE DE PAGINA

st.divider()

st.caption(
    "Proyecto de Inteligencia Artificial | "
    "Deep Learning | YOLO11 | UNAH"
)
