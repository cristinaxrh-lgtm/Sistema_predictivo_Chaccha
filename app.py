import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# ================================================================
#  CONFIGURACIÓN GLOBAL
# ================================================================

st.set_page_config(
    page_title="Chaccha Destilería — Sistema Predictivo",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paleta de colores de la marca ──────────────────────────────
COLOR_PRINCIPAL  = "#7B3F1E"   # marrón oscuro (café)
COLOR_SECUNDARIO = "#C47A3A"   # caramelo
COLOR_CLARO      = "#F2DFC8"   # crema
COLOR_LLUVIA     = "#4472C4"   # azul
COLOR_SEQUIA     = "#ED7D31"   # naranja

# ── Estilo CSS personalizado ───────────────────────────────────
st.markdown(f"""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_PRINCIPAL};
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: white !important;
        font-size: 14px;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.3);
    }}

    /* Encabezado principal */
    .encabezado {{
        background: linear-gradient(135deg, {COLOR_PRINCIPAL}, {COLOR_SECUNDARIO});
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }}
    .encabezado h2 {{
        margin: 0;
        font-size: 22px;
        font-weight: 600;
    }}
    .encabezado p {{
        margin: 4px 0 0;
        font-size: 14px;
        opacity: 0.85;
    }}

    /* Tarjetas de métricas personalizadas */
    .metric-card {{
        background: white;
        border-left: 4px solid {COLOR_PRINCIPAL};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }}
    .metric-card .valor {{
        font-size: 28px;
        font-weight: 700;
        color: {COLOR_PRINCIPAL};
    }}
    .metric-card .etiqueta {{
        font-size: 12px;
        color: #666;
        margin-top: 4px;
    }}

    /* Alerta personalizada */
    .alerta-info {{
        background: {COLOR_CLARO};
        border-left: 4px solid {COLOR_SECUNDARIO};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 14px;
        color: #4a2800;
    }}

    /* Login */
    .login-box {{
        max-width: 420px;
        margin: 4rem auto;
        background: white;
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}

    /* Badges de demanda */
    .badge-alta   {{ background:#fdecea; color:#c0392b; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }}
    .badge-media  {{ background:#fef9e7; color:#b7770d; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }}
    .badge-baja   {{ background:#eafaf1; color:#1e8449; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }}
</style>
""", unsafe_allow_html=True)


# ================================================================
#  LOGIN
# ================================================================

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    col_c, col_m, col_d = st.columns([1, 1.2, 1])
    with col_m:
        st.image("logo.png", width=160)
        st.markdown("## Sistema Predictivo de Inventarios")
        st.markdown("**Chaccha Destilería S.R.L. — Huaraz, Perú**")
        st.divider()

        usuario    = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        contraseña = st.text_input("Contraseña", type="password", placeholder="••••••••")

        if st.button("Ingresar", use_container_width=True):
            if usuario == "Administrador" and contraseña == "Chaccha2026":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

    st.stop()


# ================================================================
#  SIDEBAR
# ================================================================

with st.sidebar:
    st.image("logo.png", width=110)
    st.markdown("### Chaccha Destilería")
    st.caption("Sistema Predictivo de Inventarios")
    st.divider()

    opcion = st.radio(
        "Navegación",
        [
            "🏠  Inicio",
            "📈  Pronóstico",
            "📦  Inventario de seguridad",
            "📊  Reportes",
            "📉  Indicadores de gestión",
            "ℹ️  Acerca del sistema",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Datos históricos: 2023 – 2025")
    st.caption("Pronóstico generado: 2026")
    st.divider()

    if st.button("🔒  Cerrar sesión", use_container_width=True):
        st.session_state.login = False
        st.rerun()


# ================================================================
#  MODELOS Y DATOS
# ================================================================

modelos_config = {
    "Coca de altura":         {"hoja": "CCA", "modelo": LinearRegression(),                          "modelo_nombre": "Regresión Lineal"},
    "Muña clásica":           {"hoja": "MNC", "modelo": LinearRegression(),                          "modelo_nombre": "Regresión Lineal"},
    "Muña de altura":         {"hoja": "MNA", "modelo": MLPRegressor(max_iter=1000, random_state=42),"modelo_nombre": "Red Neuronal"},
    "Hierba luisa de altura": {"hoja": "HLA", "modelo": DecisionTreeRegressor(random_state=42),      "modelo_nombre": "Árbol de Decisión"},
    "Crema de lima":          {"hoja": "CLM",  "modelo": MLPRegressor(max_iter=1000, random_state=42),"modelo_nombre": "Red Neuronal"},
    "Eucalipto de altura":    {"hoja": "EUA", "modelo": MLPRegressor(max_iter=1000, random_state=42),"modelo_nombre": "Red Neuronal"},
}

FEATURES = ["Semana", "N° de clientes", "% ventas con descuento", "Fecha festiva", "Temporada"]
MESES    = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

@st.cache_data
def cargar_datos(hoja):
    df = pd.read_excel("datos_chaccha.xlsx", sheet_name=hoja)
    df["Fecha festiva"] = df["Fecha festiva"].map({"Sí": 1, "No": 0})
    df["Temporada"]     = df["Temporada"].map({"Lluvia": 0, "Sequía": 1})
    return df

@st.cache_resource
def entrenar_modelo(hoja, _modelo):
    df = cargar_datos(hoja)
    _modelo.fit(df[FEATURES], df["Cantidad"])
    return _modelo

def predecir_2026(hoja, _modelo):
    df     = cargar_datos(hoja)
    datos  = pd.DataFrame({
        "Semana":                  range(1, 54),
        "N° de clientes":          [df["N° de clientes"].mean()] * 53,
        "% ventas con descuento":  [df["% ventas con descuento"].mean()] * 53,
        "Fecha festiva":           [0] * 53,
        "Temporada":               [0 if s <= 13 or s >= 44 else 1 for s in range(1, 54)],
    })
    preds            = _modelo.predict(datos)
    preds            = np.maximum(0, np.round(preds)).astype(int)
    datos["Cantidad predicha"] = preds
    datos["Mes"]     = pd.cut(datos["Semana"], bins=12, labels=MESES)
    return datos

def grafico_barras(x, y, titulo, xlabel, ylabel, color=COLOR_PRINCIPAL):
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(x, y, color=color, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                    ha="center", va="bottom", fontsize=8, color="#444")
    plt.tight_layout()
    return fig

def nivel_demanda(v, p25, p75):
    if v >= p75: return "Alta"
    if v >= p25: return "Media"
    return "Baja"


# ================================================================
#  INICIO
# ================================================================

if opcion == "🏠  Inicio":

    st.markdown(f"""
    <div class="encabezado">
        <h2>Sistema Inteligente para la Gestión de Inventarios</h2>
        <p>Chaccha Destilería S.R.L. &nbsp;|&nbsp; Huaraz, Perú &nbsp;|&nbsp; Pronóstico 2026</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["6", "5", "158", "2026"],
        ["Productos tipo A", "Modelos evaluados", "Semanas analizadas", "Año de pronóstico"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="valor">{val}</div>
            <div class="etiqueta">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Modelos seleccionados por producto")
    tabla_modelos = pd.DataFrame([
        {"Producto": k, "Código": v["hoja"], "Modelo óptimo": v["modelo_nombre"]}
        for k, v in modelos_config.items()
    ])
    st.dataframe(tabla_modelos, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="alerta-info">
        <strong>¿Cómo usar el sistema?</strong><br>
        Navega desde el menú lateral. Selecciona <b>Pronóstico</b> para ver la demanda estimada 2026,
        <b>Inventario de seguridad</b> para calcular el stock mínimo recomendado,
        <b>Reportes</b> para revisar el historial de ventas e
        <b>Indicadores de gestión</b> para analizar rotación y costos.
    </div>
    """, unsafe_allow_html=True)


# ================================================================
#  PRONÓSTICO
# ================================================================


elif opcion == "📈  Pronóstico":

    st.markdown(f'<div class="encabezado"><h2>📈 Pronóstico de demanda 2026</h2><p>Predicciones generadas con el modelo óptimo por producto</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        producto = st.selectbox("Producto", list(modelos_config.keys()))
    with col2:
        vista = st.selectbox("Vista", ["Mensual", "Semanal"])
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"Modelo: **{modelos_config[producto]['modelo_nombre']}**")

    config           = modelos_config[producto]
    modelo_entrenado = entrenar_modelo(config["hoja"], config["modelo"])
    datos_2026       = predecir_2026(config["hoja"], modelo_entrenado)

    total    = datos_2026["Cantidad predicha"].sum()
    promedio = round(datos_2026["Cantidad predicha"].mean(), 1)
    sem_pico = datos_2026.loc[datos_2026["Cantidad predicha"].idxmax(), "Semana"]
    top3     = datos_2026.nlargest(3, "Cantidad predicha")["Semana"].tolist()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [total, promedio, f"Sem. {sem_pico}", modelos_config[producto]["modelo_nombre"]],
        ["Demanda proyectada total", "Promedio semanal", "Semana de mayor demanda", "Modelo utilizado"]
    ):
        col.metric(lbl, val)

    st.divider()

    if vista == "Mensual":
        resultado = datos_2026.groupby("Mes", observed=True)["Cantidad predicha"].sum().reset_index()
        fig = grafico_barras(resultado["Mes"], resultado["Cantidad predicha"],
                             f"Demanda mensual proyectada — {producto} 2026",
                             "Mes", "Unidades (botellas)")
        st.pyplot(fig)

        p25 = resultado["Cantidad predicha"].quantile(0.25)
        p75 = resultado["Cantidad predicha"].quantile(0.75)
        resultado["Nivel"] = resultado["Cantidad predicha"].apply(lambda v: nivel_demanda(v, p25, p75))

        def color_nivel(row):
            m = {"Alta": "background-color:#fdecea", "Media":"background-color:#fef9e7", "Baja":"background-color:#eafaf1"}
            return [m.get(row["Nivel"], "") if col == "Nivel" else "" for col in row.index]

        st.dataframe(resultado.style.apply(color_nivel, axis=1), use_container_width=True, hide_index=True)

    else:
        fig = grafico_barras(datos_2026["Semana"], datos_2026["Cantidad predicha"],
                             f"Demanda semanal proyectada — {producto} 2026",
                             "Semana", "Unidades (botellas)")
        st.pyplot(fig)
        st.dataframe(datos_2026[["Semana","Cantidad predicha"]], use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="alerta-info">
        ⚠️ <strong>Semanas de mayor demanda proyectada:</strong> {top3[0]}, {top3[1]} y {top3[2]}.
        Asegure stock suficiente con anticipación.
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  INVENTARIO DE SEGURIDAD
# ================================================================

elif opcion == "📦  Inventario de seguridad":

    st.markdown('<div class="encabezado"><h2>📦 Inventario de seguridad</h2><p>Cálculo del stock mínimo para garantizar el nivel de servicio deseado</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        producto = st.selectbox("Producto", list(modelos_config.keys()))
    with col2:
        st.write("")

    config = modelos_config[producto]
    df     = cargar_datos(config["hoja"])

    st.divider()
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Parámetros")
        nivel_servicio = st.slider("Nivel de servicio deseado", 85, 99, 90, 1, format="%d%%")
        plazo_entrega  = st.number_input("Plazo de entrega del proveedor (semanas)", 1, 8, 1)
        st.caption("El nivel de servicio representa la probabilidad de no tener quiebre de stock.")

    with col2:
        st.subheader("Resultados")
        demanda_media = df["Cantidad"].mean()
        desv_std      = df["Cantidad"].std()
        tabla_z       = {85:1.04, 90:1.28, 95:1.65, 96:1.75, 97:1.88, 98:2.05, 99:2.33}
        z             = tabla_z.get(nivel_servicio, 1.65)
        stock_seg     = round(z * desv_std * (plazo_entrega ** 0.5))
        pto_reorden   = round(demanda_media * plazo_entrega + stock_seg)
        inv_max       = round(demanda_media * (plazo_entrega + 4) + stock_seg)

        r1, r2 = st.columns(2)
        r1.metric("Demanda media semanal", f"{round(demanda_media)} und.")
        r2.metric("Desviación estándar",   f"{round(desv_std, 1)} und.")
        r3, r4 = st.columns(2)
        r3.metric("Stock de seguridad",    f"{stock_seg} und.")
        r4.metric("Punto de reorden",      f"{pto_reorden} und.")
        st.metric("Inventario máximo recomendado", f"{inv_max} und.")

    st.divider()
    st.markdown(f"""
    <div class="alerta-info">
        📋 <strong>Interpretación para {producto}:</strong><br>
        • Realiza un nuevo pedido cuando el inventario llegue a <strong>{pto_reorden} unidades</strong>.<br>
        • Mantén siempre al menos <strong>{stock_seg} unidades</strong> como reserva de seguridad.<br>
        • El inventario no debería superar <strong>{inv_max} unidades</strong> para evitar sobrecostos de almacenamiento.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Distribución histórica de la demanda semanal")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(df["Cantidad"], bins=20, color=COLOR_PRINCIPAL, edgecolor="white", alpha=0.85)
    ax.axvline(demanda_media, color=COLOR_SECUNDARIO, linestyle="--", linewidth=1.5, label=f"Media: {round(demanda_media)}")
    ax.axvline(pto_reorden,   color="red",            linestyle=":",  linewidth=1.5, label=f"Punto de reorden: {pto_reorden}")
    ax.set_xlabel("Unidades por semana"); ax.set_ylabel("Frecuencia")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=10)
    st.pyplot(fig)


# ================================================================
#  REPORTES
# ================================================================

elif opcion == "📊  Reportes":

    st.markdown('<div class="encabezado"><h2>📊 Reportes de demanda histórica</h2><p>Análisis del comportamiento de ventas 2023 – 2025</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        producto = st.selectbox("Producto", list(modelos_config.keys()))
    with col2:
        año_opciones = ["Todos los años", "2023", "2024", "2025"]
        año_sel      = st.selectbox("Año", año_opciones)
    with col3:
        vista_rep = st.selectbox("Vista", ["Semanal", "Mensual"])

    config = modelos_config[producto]
    df     = cargar_datos(config["hoja"])

    if año_sel != "Todos los años":
        df = df[df["Año"] == int(año_sel)]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demanda total",     f"{df['Cantidad'].sum()} und.")
    c2.metric("Promedio semanal",  f"{round(df['Cantidad'].mean())} und.")
    c3.metric("Máximo registrado", f"{df['Cantidad'].max()} und.")
    c4.metric("Semanas sin venta", f"{(df['Cantidad'] == 0).sum()}")

    st.divider()

    if vista_rep == "Semanal":
        fig, ax = plt.subplots(figsize=(12, 4))
        for anio in sorted(df["Año"].unique()):
            d = df[df["Año"] == anio]
            ax.plot(d["Semana"], d["Cantidad"], label=str(anio), marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Semana"); ax.set_ylabel("Unidades")
        ax.set_title(f"Demanda semanal — {producto}", fontsize=13, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.2, linestyle="--"); ax.legend()
        st.pyplot(fig)
    else:
        df_mes         = df.copy()
        df_mes["Mes"]  = pd.cut(df_mes["Semana"], bins=12, labels=MESES)
        por_mes        = df_mes.groupby(["Año","Mes"], observed=True)["Cantidad"].sum().reset_index()
        fig, ax        = plt.subplots(figsize=(12, 4))
        anios          = sorted(por_mes["Año"].unique())
        colores        = [COLOR_PRINCIPAL, COLOR_SECUNDARIO, "#DEB887"]
        x              = np.arange(len(MESES))
        ancho          = 0.25
        for i, anio in enumerate(anios):
            d = por_mes[por_mes["Año"] == anio]
            ax.bar(x + i*ancho, d["Cantidad"].values, ancho, label=str(anio), color=colores[i], edgecolor="white")
        ax.set_xticks(x + ancho); ax.set_xticklabels(MESES)
        ax.set_xlabel("Mes"); ax.set_ylabel("Unidades")
        ax.set_title(f"Demanda mensual — {producto}", fontsize=13, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--"); ax.legend()
        st.pyplot(fig)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Por temporada")
        df_t = df.copy()
        df_t["Temporada"] = df_t["Temporada"].map({0:"Lluvia", 1:"Sequía"})
        por_temp = df_t.groupby("Temporada")["Cantidad"].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.bar(por_temp["Temporada"], por_temp["Cantidad"],
                color=[COLOR_LLUVIA, COLOR_SEQUIA], edgecolor="white", width=0.5)
        ax2.set_ylabel("Unidades")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="y", alpha=0.2, linestyle="--")
        st.pyplot(fig2)

    with col2:
        st.subheader("Por año")
        por_anio = df.groupby("Año")["Cantidad"].sum().reset_index()
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.bar(por_anio["Año"].astype(str), por_anio["Cantidad"],
                color=COLOR_PRINCIPAL, edgecolor="white", width=0.5)
        ax3.set_ylabel("Unidades")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(axis="y", alpha=0.2, linestyle="--")
        st.pyplot(fig3)


# ================================================================
#  INDICADORES DE GESTIÓN
# ================================================================

elif opcion == "📉  Indicadores de gestión":

    st.markdown('<div class="encabezado"><h2>📉 Indicadores de gestión de inventarios</h2><p>Rotación, costos y análisis comparativo histórico vs. proyectado 2026</p></div>', unsafe_allow_html=True)

    try:
        rotacion = pd.read_excel("indicadores_chaccha.xlsx", sheet_name="Rotacion")
        costos_p = pd.read_excel("indicadores_chaccha.xlsx", sheet_name="Costos")
    except Exception as e:
        st.error(f"No se pudo cargar indicadores_chaccha.xlsx: {e}")
        st.stop()

    rotacion["Producto"] = rotacion["Producto"].str.strip()
    costos_p["Producto"] = costos_p["Producto"].str.strip()

    rotacion["Codigo"] = rotacion["Codigo"].str.strip()
    costos_p["Codigo"] = costos_p["Codigo"].str.strip()

    # Unir tablas
    df = rotacion.merge(
        costos_p[
            [
                "Codigo",
                "Costo_almacenamiento",
                "Costo_quiebre"
            ]
        ],
        on="Codigo",
        how="left"
    )

    # Cálculos
    df["Inventario_promedio"] = (
        df["Stock_inicial"] + df["Stock_final"]
    ) / 2

    df["Valor_inventario_promedio"] = (
        df["Inventario_promedio"] * df["Costo_unitario"]
    )

    df["Costo_ventas"] = (
        df["Ventas_mes"] * df["Costo_unitario"]
    )

    df["Rotacion_inventario"] = (
        df["Costo_ventas"] / df["Valor_inventario_promedio"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    df["Costo_almacenamiento_mes"] = (
        df["Valor_inventario_promedio"] *
        (df["Costo_almacenamiento"] / 12)
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        producto = st.selectbox("Producto", df["Producto"].unique())
    with col2:
        año_ind  = st.selectbox("Año", ["Todos", "2023", "2024", "2025"])
    with col3:
        vista_ind = st.selectbox("Período", ["Histórico", "Proyectado 2026"])

    datos_p = df[df["Producto"] == producto]
    if año_ind != "Todos":
        datos_p = datos_p[datos_p["Año"] == int(año_ind)]

    st.divider()

    if vista_ind == "Histórico":

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rotación promedio",    f"{datos_p['Rotacion_inventario'].mean():.2f} veces")
        c2.metric("Inventario promedio",  f"{datos_p['Inventario_promedio'].mean():.0f} und.")
        c3.metric("Valor inventario",     f"S/ {datos_p['Valor_inventario_promedio'].mean():,.2f}")
        c4.metric("Costo almacenamiento", f"S/ {datos_p['Costo_almacenamiento_mes'].sum():,.2f}")

        st.divider()
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Rotación mensual")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colores_anio = [COLOR_PRINCIPAL, COLOR_SECUNDARIO, "#DEB887"]
            for i, anio in enumerate(sorted(datos_p["Año"].unique())):
                d = datos_p[datos_p["Año"] == anio]
                ax.plot(d["Mes"], d["Rotacion_inventario"], label=str(anio),
                        marker="o", color=colores_anio[i % 3])
            ax.set_xlabel("Mes"); ax.set_ylabel("Rotación")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(alpha=0.2, linestyle="--"); ax.legend()
            st.pyplot(fig)

        with col_g2:
            st.subheader("Costo de almacenamiento mensual")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            for i, anio in enumerate(sorted(datos_p["Año"].unique())):
                d = datos_p[datos_p["Año"] == anio]
                ax2.bar(d["Mes"].astype(str), d["Costo_almacenamiento_mes"],
                        label=str(anio), alpha=0.8, color=colores_anio[i % 3])
            ax2.set_xlabel("Mes"); ax2.set_ylabel("S/.")
            ax2.spines[["top","right"]].set_visible(False)
            ax2.grid(axis="y", alpha=0.2, linestyle="--"); ax2.legend()
            st.pyplot(fig2)

        st.divider()
        st.subheader("Detalle mensual")
        tabla = datos_p[["Año","Mes","Ventas_mes","Inventario_promedio",
                          "Rotacion_inventario","Costo_almacenamiento_mes"]].copy()
        tabla.columns = ["Año","Mes","Ventas","Inv. promedio","Rotación","Costo almac."]
        tabla["Ventas"]        = tabla["Ventas"].round(0).astype(int)
        tabla["Inv. promedio"] = tabla["Inv. promedio"].round(0).astype(int)
        tabla["Rotación"]      = tabla["Rotación"].round(2)
        tabla["Costo almac."]  = tabla["Costo almac."].apply(lambda x: f"S/ {x:,.2f}")
        st.dataframe(tabla, use_container_width=True, hide_index=True)

    else:
        # ── Proyectado 2026 ──────────────────────────────────────
        config_p         = modelos_config[producto]
        modelo_p         = entrenar_modelo(config_p["hoja"], config_p["modelo"])
        datos_2026_p     = predecir_2026(config_p["hoja"], modelo_p)
        por_mes_2026     = datos_2026_p.groupby("Mes", observed=True)["Cantidad predicha"].sum().reset_index()

        costo_unit       = costos_p.loc[costos_p["Producto"] == producto, "Costo_unitario"].values[0]
        costo_alm_pct    = costos_p.loc[costos_p["Producto"] == producto, "Costo_almacenamiento"].values[0]

        df_proy = cargar_datos(config_p["hoja"])
        desv_p  = df_proy["Cantidad"].std()
        ss_p    = round(1.65 * desv_p)

        por_mes_2026["Inv_promedio_proy"]   = (por_mes_2026["Cantidad predicha"] / 2 + ss_p).round(0).astype(int)
        por_mes_2026["Valor_inv_proy"]      = por_mes_2026["Inv_promedio_proy"] * costo_unit
        por_mes_2026["Costo_alm_proy"]      = por_mes_2026["Valor_inv_proy"] * (costo_alm_pct / 12)
        por_mes_2026["Rotacion_proy"]       = (por_mes_2026["Cantidad predicha"] * costo_unit / por_mes_2026["Valor_inv_proy"]).round(2)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rotación proyectada",       f"{por_mes_2026['Rotacion_proy'].mean():.2f} veces")
        c2.metric("Inv. promedio proyectado",  f"{round(por_mes_2026['Inv_promedio_proy'].mean())} und.")
        c3.metric("Valor inventario proy.",    f"S/ {por_mes_2026['Valor_inv_proy'].mean():,.2f}")
        c4.metric("Costo almac. proyectado",   f"S/ {por_mes_2026['Costo_alm_proy'].sum():,.2f}")

        st.divider()
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Demanda proyectada 2026")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(por_mes_2026["Mes"], por_mes_2026["Cantidad predicha"],
                   color=COLOR_PRINCIPAL, edgecolor="white")
            ax.set_ylabel("Unidades")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            st.pyplot(fig)

        with col_g2:
            st.subheader("Costo de almacenamiento proyectado")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            ax2.bar(por_mes_2026["Mes"], por_mes_2026["Costo_alm_proy"],
                    color=COLOR_SECUNDARIO, edgecolor="white")
            ax2.set_ylabel("S/.")
            ax2.spines[["top","right"]].set_visible(False)
            ax2.grid(axis="y", alpha=0.2, linestyle="--")
            st.pyplot(fig2)

        st.divider()
        st.subheader("Detalle mensual proyectado 2026")
        tabla_p = por_mes_2026[["Mes","Cantidad predicha","Inv_promedio_proy","Rotacion_proy","Costo_alm_proy"]].copy()
        tabla_p.columns = ["Mes","Demanda proyectada","Inv. promedio","Rotación","Costo almac."]
        tabla_p["Costo almac."] = tabla_p["Costo almac."].apply(lambda x: f"S/ {x:,.2f}")
        st.dataframe(tabla_p, use_container_width=True, hide_index=True)


# ================================================================
#  ACERCA DEL SISTEMA
# ================================================================

elif opcion == "ℹ️  Acerca del sistema":

    st.markdown('<div class="encabezado"><h2>ℹ️ Acerca del sistema</h2><p>Información técnica y metodológica del sistema predictivo</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Descripción general")
        st.markdown("""
        Sistema de apoyo a la toma de decisiones para la gestión de inventarios de
        **Chaccha Destilería S.R.L.**, desarrollado como parte del proyecto de tesis
        de pregrado en Ingeniería Industrial — UNASAM 2026.

        Emplea modelos de **machine learning** entrenados con datos históricos de ventas
        (2023–2025) para generar pronósticos de demanda semanal y mensual para el año 2026,
        integrando el cálculo del inventario de seguridad y los indicadores de gestión logística.
        """)

        st.subheader("Funcionalidades")
        st.markdown("""
        - Pronóstico de demanda semanal y mensual por producto
        - Cálculo de stock de seguridad y punto de reorden
        - Análisis histórico de ventas con filtros por año
        - Indicadores de rotación y costos de inventario
        - Proyección de indicadores para 2026
        """)

    with col2:
        st.subheader("Modelos predictivos")
        tabla_mod = pd.DataFrame([
            {"Producto": k, "Modelo": v["modelo_nombre"]}
            for k, v in modelos_config.items()
        ])
        st.dataframe(tabla_mod, use_container_width=True, hide_index=True)

        st.subheader("Herramientas utilizadas")
        st.markdown("""
        | Herramienta | Uso |
        |---|---|
        | Python 3.13 | Lenguaje de programación |
        | Streamlit | Interfaz web |
        | Scikit-learn | Modelos de machine learning |
        | Pandas / NumPy | Procesamiento de datos |
        | Matplotlib | Visualización |
        | Orange | Evaluación y selección de modelos |
        """)

    st.divider()
    st.caption("Desarrollado por: Cristina Ximena Ramirez Huatuco")
    