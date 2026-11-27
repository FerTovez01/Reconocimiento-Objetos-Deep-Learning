
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import av
import json
import threading
import os
import time
import torch

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from twilio.rest import Client
from collections import Counter

# ==========================================
# CONFIGURACION DE LA PAGINA
# ==========================================

st.set_page_config(
    page_title="Reconocimiento de Objetos",
    page_icon="📷",
    layout="wide"
)

st.title("Reconocimiento de Objetos con Deep Learning")

st.markdown("""
Sistema inteligente de reconocimiento de objetos
mediante una red neuronal profunda YOLO11 preentrenada.
""")

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    modelo = YOLO("yolo11n.pt")

    return modelo, threading.Lock()

modelo, bloqueo = cargar_modelo()

dispositivo = 0 if torch.cuda.is_available() else "cpu"

# ==========================================
# CLASES EN ESPAÑOL
# ==========================================

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

# ==========================================
# METRICAS DE EVALUACION
# ==========================================

st.subheader("Rendimiento del modelo")

metricas = {}

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
        f"Dataset: {metricas.get('dataset', 'COCO128')} | "
        f"Resolucion evaluada: {metricas.get('imgsz', 640)}"
    )

    st.caption(
        "mAP es la metrica principal de deteccion "
        "y no equivale al accuracy tradicional."
    )

    if metricas.get("dataset", "").lower() == "coco128":

        st.caption(
            "Evaluacion demostrativa sobre COCO128. "
            "No constituye una prueba independiente."
        )

else:

    st.warning("No se encontro metricas.json")

# ==========================================
# MENU LATERAL
# ==========================================

st.sidebar.title("Configuracion")

st.sidebar.success("Modelo YOLO11n cargado")

st.sidebar.metric(
    "Categorias disponibles",
    len(clases_es)
)

st.sidebar.write(
    "Procesador:",
    "GPU" if torch.cuda.is_available() else "CPU"
)

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
        "Capturar fotografia",
        "Evaluacion del modelo",
        "Informacion del modelo"
    ]
)

# ==========================================
# CONFIGURACION TURN - TWILIO
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def obtener_servidores_ice():

    cliente = Client(
        st.secrets["twilio"]["account_sid"],
        st.secrets["twilio"]["auth_token"]
    )

    token = cliente.tokens.create(ttl=3600)

    servidores = []

    for servidor in token.ice_servers:

        urls = servidor.get("urls") or servidor.get("url")

        if not urls:
            continue

        configuracion = {
            "urls": urls
        }

        if servidor.get("username"):
            configuracion["username"] = servidor["username"]

        if servidor.get("credential"):
            configuracion["credential"] = servidor["credential"]

        servidores.append(configuracion)

    return servidores

# ==========================================
# FUNCION DE DETECCION
# ==========================================

def detectar_objetos(imagen, umbral, tamano):

    with bloqueo:

        resultados = modelo.predict(
            source=imagen,
            conf=umbral,
            imgsz=tamano,
            device=dispositivo,
            half=torch.cuda.is_available(),
            verbose=False
        )

    imagen_resultado = imagen.copy()

    objetos = []

    for resultado in resultados:

        for caja in resultado.boxes:

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0].tolist()
            )

            clase = int(caja.cls.item())

            probabilidad = float(caja.conf.item())

            nombre = clases_es.get(
                clase,
                str(clase)
            )

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
                    probabilidad * 100,
                    2
                )
            })

    return imagen_resultado, objetos

# ==========================================
# MOSTRAR RESULTADOS
# ==========================================

def mostrar_resultados(objetos):

    st.metric(
        "Total de objetos detectados",
        len(objetos)
    )

    if objetos:

        df = pd.DataFrame(objetos)

        st.subheader("Resultados de deteccion")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.metric(
            "Confianza promedio",
            f"{df['Confianza'].mean():.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Confianza por objeto")

            st.bar_chart(
                df.groupby("Objeto")["Confianza"].mean()
            )

        with col2:

            st.subheader("Cantidad por categoria")

            conteo = Counter(
                objeto["Objeto"]
                for objeto in objetos
            )

            st.bar_chart(pd.Series(conteo))

        st.download_button(
            label="Descargar resultados CSV",
            data=df.to_csv(index=False),
            file_name="resultados.csv",
            mime="text/csv"
        )

    else:

        st.warning("No se detectaron objetos")

# ==========================================
# OPCION 1: CARGAR IMAGEN
# ==========================================

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

            mostrar_resultados(objetos)

# ==========================================
# OPCION 2: CAMARA EN TIEMPO REAL
# ==========================================

elif opcion == "Camara en tiempo real":

    st.subheader("Reconocimiento en tiempo real")

    st.info(
        "Presiona START y permite el acceso "
        "a la camara de tu computadora."
    )

    # Obtener servidores STUN y TURN

    try:

        servidores_ice = obtener_servidores_ice()

        tiene_turn = False

        for servidor in servidores_ice:

            urls = servidor["urls"]

            if isinstance(urls, str):
                urls = [urls]

            if any(
                url.startswith(("turn:", "turns:"))
                for url in urls
            ):
                tiene_turn = True

        if tiene_turn:

            st.success(
                "Credenciales TURN obtenidas correctamente"
            )

        else:

            st.warning(
                "No se recibieron servidores TURN"
            )

    except Exception:

        tiene_turn = False

        servidores_ice = [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                    "stun:stun2.l.google.com:19302"
                ]
            }
        ]

        st.warning(
            "TURN no disponible. "
            "Verifica las credenciales de Streamlit Secrets."
        )

    # Procesar video

    def procesar_video(frame):

        imagen = frame.to_ndarray(
            format="bgr24"
        )

        inicio = time.perf_counter()

        resultado, objetos = detectar_objetos(
            imagen,
            confianza,
            resolucion
        )

        tiempo = time.perf_counter() - inicio

        fps = 1 / max(tiempo, 0.000001)

        cv2.putText(
            resultado,
            f"Objetos: {len(objetos)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            resultado,
            f"FPS inferencia: {fps:.1f}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            resultado,
            format="bgr24"
        )

    # Activar camara WebRTC

    contexto = webrtc_streamer(

        key="reconocimiento-yolo11-turn",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=procesar_video,

        rtc_configuration={
            "iceServers": servidores_ice
        },

        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},
                "height": {"ideal": 360},
                "frameRate": {
                    "ideal": 15,
                    "max": 20
                }
            },
            "audio": False
        },

        async_processing=True
    )

    if contexto.state.playing:

        st.success("Camara conectada")

    st.caption(
        "La velocidad depende de la conexion "
        "y los recursos del servidor."
    )

# ==========================================
# OPCION 3: CAPTURAR FOTOGRAFIA
# ==========================================

elif opcion == "Capturar fotografia":

    st.subheader("Reconocimiento mediante fotografia")

    fotografia = st.camera_input(
        "Tomar fotografia"
    )

    if fotografia is not None:

        bytes_imagen = np.frombuffer(
            fotografia.getvalue(),
            dtype=np.uint8
        )

        imagen = cv2.imdecode(
            bytes_imagen,
            cv2.IMREAD_COLOR
        )

        if imagen is None:

            st.error("No se pudo procesar la fotografia")

            st.stop()

        with st.spinner("Reconociendo objetos..."):

            resultado, objetos = detectar_objetos(
                imagen,
                confianza,
                resolucion
            )

        st.image(
            resultado,
            channels="BGR",
            caption="Objetos reconocidos"
        )

        mostrar_resultados(objetos)

# ==========================================
# OPCION 4: EVALUACION DEL MODELO
# ==========================================

elif opcion == "Evaluacion del modelo":

    st.subheader("Evaluacion de la red neuronal")

    if metricas:

        df_metricas = pd.DataFrame({
            "Metrica": [
                "Precision",
                "Recall",
                "mAP50",
                "mAP50-95"
            ],
            "Porcentaje": [
                metricas["Precision"],
                metricas["Recall"],
                metricas["mAP50"],
                metricas["mAP50-95"]
            ]
        })

        st.dataframe(
            df_metricas,
            use_container_width=True
        )

        st.bar_chart(
            df_metricas.set_index("Metrica")
        )

        st.markdown("""
        ### Interpretacion de resultados

        **Precision:** proporcion de detecciones correctas.

        **Recall:** proporcion de objetos reales detectados.

        **mAP50:** precision promedio con IoU de 0.50.

        **mAP50-95:** precision promedio utilizando
        diferentes umbrales IoU.

        **Confianza:** puntuacion asignada a cada
        deteccion individual.
        """)

    else:

        st.warning("No hay metricas disponibles")

# ==========================================
# OPCION 5: INFORMACION DEL MODELO
# ==========================================

elif opcion == "Informacion del modelo":

    st.subheader("Arquitectura de Deep Learning")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Modelo:** YOLO11n")
        st.write("**Framework:** PyTorch")
        st.write("**Dataset:** COCO")
        st.write("**Categorias:** 80")

    with col2:

        st.write("**Tipo:** Red neuronal profunda")
        st.write("**Tarea:** Deteccion de objetos")
        st.write("**Procesamiento:** OpenCV")
        st.write("**Interfaz:** Streamlit")

    st.subheader("Categorias disponibles")

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

# ==========================================
# PIE DE PAGINA
# ==========================================

st.divider()

st.caption(
    "Proyecto de Inteligencia Artificial | "
    "Deep Learning | YOLO11 | UNAH"
)
