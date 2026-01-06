import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(page_title="Reliarisk StatX", layout="wide")

st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stTextArea textarea { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCIONES DE AYUDA (CALLBACKS)
# ==========================================
# Esta función se ejecuta ANTES de recargar la app cuando se pulsa Reiniciar
def clear_input_callback():
    st.session_state.data_input = ""

# ==========================================
# LOGO Y TÍTULO
# ==========================================
if os.path.exists("mi_logo.png"):
    st.sidebar.image("mi_logo.png", width=200, caption="Grupo Reliarisk")

st.title("Reliarisk StatX")
st.markdown("**Plataforma de Caracterización Estadística Avanzada** | *Versión 2.2*")
st.markdown("---")

# ==========================================
# 1. INGESTA DE DATOS
# ==========================================
st.sidebar.header("1. Configuración de Datos")

st.sidebar.markdown("### 📋 Carga de Datos")
st.sidebar.caption("Pegue los valores de su variable (uno por línea o separados por comas/espacios).")

# Widget de entrada de texto
# Nota: 'key="data_input"' vincula este cuadro al session_state
raw_text_input = st.sidebar.text_area("Valores de la Variable:", height=200, key="data_input")

# Opciones adicionales
force_loc_zero = st.sidebar.checkbox("Forzar origen en cero (Loc=0)", value=True, 
    help="Activar para emular el comportamiento de 2 parámetros (Weibull, Gamma, Lognormal comenzarán estrictamente en 0).")

# Función para procesar el texto pegado
def process_text_data(text_data):
    if not text_data.strip():
        return None
    try:
        cleaned_text = text_data.replace(',', ' ').replace('\n', ' ')
        values = [float(x) for x in cleaned_text.split() if x.strip()]
        if not values:
            return None
        return pd.DataFrame(values, columns=["Valor"])
    except ValueError:
        st.sidebar.error("Error: Asegúrese de pegar solo valores numéricos válidos.")
        return None

# Botones de Control
col_btn1, col_btn2 = st.sidebar.columns(2)

with col_btn1:
    start_analysis = st.button("▶ Iniciar Análisis", use_container_width=True, type="primary")

with col_btn2:
    # CORRECCIÓN: Usamos on_click para limpiar el estado de forma segura
    st.button("🔄 Reiniciar Cálculo", use_container_width=True, on_click=clear_input_callback)

# Lógica Principal
df = None
# Si se presiona iniciar y hay texto, o si ya se inició previamente (persistencia simple)
if start_analysis and raw_text_input:
    df = process_text_data(raw_text_input)
elif raw_text_input and not start_analysis:
    st.info("ℹ️ Haga clic en '▶ Iniciar Análisis' en la barra lateral para procesar los datos pegados.")


if df is not None and start_analysis:
    col_name = "Variable Pegada"
    
    # Limpieza de datos
    data = df[df.columns[0]].dropna()
    data = data[np.isfinite(data)] 
    data = data[data > 0] if force_loc_zero else data 
    
    if len(data) < 5:
        st.error("Error: Se necesitan al menos 5 valores válidos para un análisis fiable.")
        st.stop()

    with st.expander(f"🔍 Verificación de Datos (N={len(data)})"):
        st.dataframe(data.to_frame().T, height=150)

    # ==========================================
    # 2. MOTOR DE CÁLCULO
    # ==========================================
    
    dist_names = ['norm', 'lognorm', 'weibull_min', 'expon', 'gamma', 'uniform', 'beta']
    results = []

    progress_text = "Ajustando distribuciones..."
    my_bar = st.progress(0, text=progress_text)

    for i, name in enumerate(dist_names):
        dist = getattr(stats, name)
        
        try:
            # Lógica de Ajuste (Fit)
            if force_loc_zero and name in ['weibull_min', 'gamma', 'lognorm', 'expon', 'beta']:
                 params = dist.fit(data, floc=0)
            else:
                 params = dist.fit(data)
                
            # Formateo de parámetros
            param_str = ""
            if name == 'norm':
                param_str = f"Media={params[0]:.2f}, Desv={params[1]:.2f}"
            elif name == 'weibull_min':
                param_str = f"Forma={params[0]:.2f}, Escala={params[2]:.2f}, Loc={params[1]:.2f}"
            elif name == 'lognorm':
                s, loc, scale = params
                median_val = scale
                mu_log = np.log(scale)
                param_str = f"DSt={s:.3f}, Mediana={median_val:.3f} (Mediat={mu_log:.3f})"
            elif name == 'expon':
                param_str = f"Loc={params[0]:.2f}, Escala={params[1]:.2f}"
            elif name == 'gamma':
                param_str = f"Alpha={params[0]:.2f}, Beta={params[2]:.2f}, Loc={params[1]:.2f}"
            elif name == 'beta':
                param_str = f"Alpha={params[0]:.2f}, Beta={params[1]:.2f}, Min={params[2]:.2f}, Max={params[2]+params[3]:.2f}"
            else:
                param_str = ", ".join([f"{p:.2f}" for p in params])

            # Cálculo Anderson-Darling (A2)
            n = len(data)
            sorted_data = np.sort(data)
            cdf_vals = np.clip(dist.cdf(sorted_data, *params), 1e-10, 1 - 1e-10)
            
            s_val = np.sum((2*np.arange(1, n+1) - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
            ad_stat = -n - s_val/n
            
            ks_stat, ks_p = stats.kstest(data, name, args=params)

            results.append({
                "Distribución": name.capitalize(),
                "Estadístico AD": ad_stat,
                "Valor P (KS)": ks_p,
                "Parámetros Detectados": param_str,
                "Object": dist,
                "Params": params
            })
        except Exception as e:
            pass
        
        my_bar.progress((i + 1) / len(dist_names))

    my_bar.empty()
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="Estadístico AD", ascending=True).reset_index(drop=True)

        # ==========================================
        # 3. INTERFAZ DE RESULTADOS
        # ==========================================
        
        col_master_1, col_master_2 = st.columns([4, 6])

        with col_master_1:
            st.subheader("Tabla de Resultados")
            st.dataframe(
                results_df[["Distribución", "Estadístico AD", "Valor P (KS)", "Parámetros Detectados"]].style.background_gradient(subset=["Estadístico AD"], cmap="Greens_r"),
                use_container_width=True
            )
            
            st.info("💡 Nota: Un 'Estadístico AD' más bajo indica un mejor ajuste.")

            st.markdown("### 🛠️ Análisis Detallado")
            selected_dist_name = st.selectbox(
                "Seleccione la distribución a visualizar:", 
                results_df["Distribución"].tolist()
            )
            
            selected_row = results_df[results_df["Distribución"] == selected_dist_name].iloc[0]
            sel_dist = selected_row["Object"]
            sel_params = selected_row["Params"]

            st.markdown("---")
            st.markdown("### 🧮 Calculadora de Probabilidad")
            
            calc_tab1, calc_tab2 = st.tabs(["Valor ⮕ Percentil", "Percentil ⮕ Valor"])
            
            with calc_tab1:
                val_input = st.number_input(f"Ingresar valor:", value=float(np.mean(data)))
                perc_result = sel_dist.cdf(val_input, *sel_params) * 100
                st.metric("Percentil (Prob. Acumulada)", f"{perc_result:.2f}%")
                
            with calc_tab2:
                perc_input = st.number_input("Ingresar Percentil (0-100%):", value=50.0, min_value=0.01, max_value=99.99)
                val_result = sel_dist.ppf(perc_input/100, *sel_params)
                st.metric(f"Valor estimado para P{perc_input:.0f}", f"{val_result:.4f}")

        with col_master_2:
            st.subheader(f"Visualización: {selected_dist_name}")
            
            plot_type = st.radio(
                "Tipo de Gráfico:",
                ["Densidad (PDF)", "Acumulada (CDF)", "Acumulada Inversa (1-CDF)"],
                horizontal=True
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_min, x_max = min(data), max(data)
            pad = (x_max - x_min) * 0.1
            x_plot = np.linspace(max(0, x_min - pad) if force_loc_zero else x_min - pad, x_max + pad, 1000)
            
            if plot_type == "Densidad (PDF)":
                sns.histplot(data, stat="density", bins='auto', color="#87CEEB", alpha=0.5, label="Datos Reales", ax=ax)
                y_plot = sel_dist.pdf(x_plot, *sel_params)
                ax.plot(x_plot, y_plot, 'r-', lw=2.5, label=f"Ajuste {selected_dist_name}")
                ax.set_title("Función de Densidad de Probabilidad")
                ax.set_ylabel("Densidad")
                
            elif plot_type == "Acumulada (CDF)":
                sns.histplot(data, stat="density", bins='auto', cumulative=True, element="step", fill=False, color="gray", label="Datos Empíricos", ax=ax)
                y_plot = sel_dist.cdf(x_plot, *sel_params)
                ax.plot(x_plot, y_plot, 'g-', lw=2.5, label=f"CDF {selected_dist_name}")
                ax.set_title("Función de Distribución Acumulada")
                ax.set_ylabel("Probabilidad Acumulada")
                
            elif plot_type == "Acumulada Inversa (1-CDF)":
                sorted_data_inv = np.sort(data)
                y_emp = 1.0 - np.arange(1, len(data)+1) / len(data)
                ax.step(sorted_data_inv, y_emp, where='post', color='gray', label='Datos Empíricos (Survival)')
                
                y_plot = sel_dist.sf(x_plot, *sel_params) 
                ax.plot(x_plot, y_plot, 'purple', lw=2.5, label=f"1-CDF {selected_dist_name}")
                ax.set_title("Función de Supervivencia")
                ax.set_ylabel("Probabilidad (1 - P)")

            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            with st.expander("Ver Gráfico P-P (Diagnóstico de Linealidad)"):
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                stats.probplot(data, dist=sel_dist, sparams=sel_params, plot=ax2, fit=False)
                ax2.plot([min(data), max(data)], [min(data), max(data)], 'r--', lw=2)
                st.pyplot(fig2)
    else:
         st.warning("No se pudo ajustar ninguna distribución a los datos proporcionados.")

else:
    if not raw_text_input:
         st.info("👋 Por favor, pegue sus datos en la barra lateral para comenzar.")
