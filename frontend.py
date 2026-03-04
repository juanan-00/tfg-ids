import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import plotly.express as px

BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title = "IDS ML Dashboard",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

#abrir archivo .css
with open("style.css") as f:
    css_file = f.read()

st.markdown(f"<style>{css_file}</style>", unsafe_allow_html=True)

@st.cache_data(ttl=5)
def get_api_health():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_interfaces():
    """Obtiene las interfaces de red disponibles desde la API"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/interfaces", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def cambiar_interface(interface_name):
    """Cambia la interfaz de red activa"""
    try:
        response = requests.post(f"{BASE_URL}/api/v1/interfaces/{interface_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def escanear_trafico_red():
    try:
        response = requests.post(f"{BASE_URL}/start_escaner", timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"error": "no se pudo realizar la peticion correctamente"}

def stop_trafico_red():
    try:
        response = requests.post(f"{BASE_URL}/stop_escaner", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"error": "no se pudo realizar la peticion correctamente"}

@st.cache_data(ttl=5)
def obtener_csv():
    try:
        response = requests.get(f"{BASE_URL}/api/v1/last", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                df = pd.DataFrame(data['data'])
                return df, True
            return pd.DataFrame(), False
    except:
        pass
    return pd.DataFrame(), False

def analizar_pcap(pcap_file):
    """
    Envía un archivo PCAP a la API para ser analizado y convertido a CSV
    """
    try:
        # Usar application/octet-stream para archivos PCAP
        files = {"file": (pcap_file.name, pcap_file, "application/octet-stream")}
        response = requests.post(f"{BASE_URL}/analizar_pcap", files=files, timeout=120)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error en la API: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error al enviar archivo: {str(e)}"}

# ==================== PÁGINAS ====================

def obtener_threats():
    """Obtiene las amenazas del último batch"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/threats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                return pd.DataFrame(data['data']), data.get('rows', 0)
    except:
        pass
    return pd.DataFrame(), 0

def pagina_monitoreo():
    """Página de monitoreo en tiempo real"""
    st.title("Monitoreo en Tiempo Real")
    st.header("Datos de la sesión")
    health = get_api_health()
    stop_loop = False
    col1, col2, col3 = st.columns(3)

    with col1:
        if health:
            status = "Online" if health.get('status') == 'ok' else health.get('status') == 'no'
            st.metric("Estado", status)
    with col2:
        if health:
            st.metric("Interfaz de red", health.get('interfaz'))
    with col3:
        if health:
            iface_data = get_interfaces()
            if iface_data:
                current_iface = next((i for i in iface_data["interfaces"] if i["name"] == health.get('interfaz')), {})
                st.metric("IP", current_iface.get("ip", "Sin IP"))

    st.markdown("---")

    st.header("Iniciar captura tráfico red")
    col1, col2 = st.columns([1,1], gap="small")

    with col1:
        if st.button("Iniciar captura", type="primary", use_container_width=True):
            res_peticion = escanear_trafico_red()
            st.json(res_peticion)
    with col2:
        if st.button("Detener captura", type="primary", use_container_width=True):
            res_pet_stop = stop_trafico_red()
            st.json(res_pet_stop)
            stop_loop = True

    st.markdown("---")

    # Panel de alertas
    st.header("Alertas recientes")
    alerts_count_placeholder = st.empty()
    alerts_placeholder = st.empty()

    st.markdown("---")

    # Predicciones a ancho completo
    st.header("Predicciones del ultimo batch")
    status_placeholder = st.empty()
    df_placeholder = st.empty()

    st.markdown("---")

    # Logs del sistema en expander
    with st.expander("Logs del sistema", expanded=False):
        logs_placeholder = st.empty()

    # Loop de actualización automática
    while True:
        if stop_loop:
            df, status = obtener_csv()
            df_placeholder.dataframe(data=df, use_container_width=True)
            return

        # Predicciones
        try:
            df, success = obtener_csv()
            if success:
                status_placeholder.success(f"Ultimo batch {datetime.now().strftime('%H:%M:%S')}")
                df_placeholder.dataframe(data=df, use_container_width=True)
            else:
                status_placeholder.error("Fallo al ejecutar la peticion")
                df_placeholder.empty()
        except Exception as e:
            status_placeholder.error(f"Error: {e}")
            df_placeholder.empty()

        # Alertas
        try:
            threats_df, threats_count = obtener_threats()
            if threats_count > 0:
                alerts_count_placeholder.error(f"Se han detectado {threats_count} amenazas")
                display_cols = []
                for col in ['prediccion', 'confianza', 'src_ip', 'dst_ip',
                             'src_port', 'dst_port', 'protocol',
                             'FlowDuration', 'TotalFwdPackets',
                             'TotalBackwardPackets', 'tiempo']:
                    if col in threats_df.columns:
                        display_cols.append(col)
                if not display_cols:
                    display_cols = threats_df.columns.tolist()[:8]
                alerts_placeholder.dataframe(
                    threats_df[display_cols],
                    use_container_width=True
                )
            else:
                alerts_count_placeholder.success("Sin amenazas detectadas")
                alerts_placeholder.empty()
        except Exception:
            pass

        # Logs
        try:
            logs_response = requests.get(f"{BASE_URL}/api/v1/logs?last=20", timeout=5)
            if logs_response.status_code == 200:
                logs_data = logs_response.json()
                log_entries = logs_data.get("logs", [])
                if log_entries:
                    log_text = "\n".join(
                        f"{entry['timestamp'][:19]}  {entry['message']}"
                        for entry in reversed(log_entries)
                    )
                    logs_placeholder.code(log_text, language="log")
        except Exception:
            pass

        time.sleep(2)

def pagina_analisis_csv():
    """Página para analizar archivos CSV y PCAP"""
    st.title("Análisis de archivos pcap y csv")

    # Opción para cargar archivo o usar el último
    opcion = st.radio(
        "Selecciona el origen de datos:",
        ["Analizar archivo CSV", "Analizar archivo PCAP"]
    )

    df = None

    if opcion == "Analizar archivo CSV":
        uploaded_file = st.file_uploader("Sube un archivo CSV", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Archivo subido correctamente")

    else:  # Analizar archivo PCAP
        st.info("Proceso: PCAP -> Zeek -> CSV -> Predicciones UNSW-NB15")
        uploaded_pcap = st.file_uploader("Sube un archivo PCAP", type=['pcap', 'pcapng'], key="pcap_uploader")
        if uploaded_pcap is not None:
            # Crear expander para mostrar el proceso
            with st.expander("Proceso de análisis", expanded=True):
                st.write("1. Subiendo archivo PCAP...")
                st.write("2. Procesando con Zeek...")
                st.write("3. Generando CSV desde logs de Zeek...")
                st.write("4. Visualizando resultados...")

            with st.spinner("Procesando archivo PCAP..."):
                resultado = analizar_pcap(uploaded_pcap)

                if "error" in resultado:
                    st.error(f"{resultado['error']}")
                elif resultado.get("status") == "success":
                    st.success(f"Archivo PCAP procesado exitosamente: {resultado.get('filename')}")

                    # Mostrar información del CSV generado
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CSV Generado", resultado.get('csv_generated'))
                    with col2:
                        st.metric("Total de Filas", resultado.get('rows'))

                    st.info("Visualizando CSV generado a continuacion...")

                    # Convertir los datos a DataFrame
                    df = pd.DataFrame(resultado['data'])
                else:
                    st.error("Error al procesar el archivo")

    # Mostrar análisis si hay dataframe
    if df is not None:
        pred_cols = [col for col in df.columns if 'pred' in col.lower()]
        pred_col = pred_cols[0] if pred_cols else None

        if pred_col:
            st.markdown("---")
            st.subheader("Estadisticas por tipo de conexion")

            total = len(df)
            value_counts = df[pred_col].value_counts()

            stats_rows = []
            for tipo in value_counts.index:
                subset = df[df[pred_col] == tipo]
                row = {
                    "Tipo": tipo,
                    "Cantidad": len(subset),
                    "Porcentaje": f"{len(subset) / total * 100:.1f}%",
                }
                if 'confianza' in df.columns:
                    row["Confianza media"] = f"{subset['confianza'].mean():.2%}"
                stats_rows.append(row)

            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Vista previa de datos")
        st.dataframe(df, use_container_width=True)

def pagina_historial_sesiones():
    """Página de historial de sesiones de escaneo"""
    st.title("Historial de Sesiones")

    def formato_fecha(fecha_str):
        if not fecha_str:
            return "-"
        try:
            dt = datetime.fromisoformat(fecha_str)
            return dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            return fecha_str[:19]

    try:
        response = requests.get(f"{BASE_URL}/api/v1/sesiones", timeout=10)
        if response.status_code == 200:
            data = response.json()
            sesiones = data.get("sesiones", [])

            if sesiones:
                # Tabla resumen de sesiones
                df_sesiones = pd.DataFrame(sesiones)
                df_display = df_sesiones[["id", "fecha_inicio", "fecha_fin", "total_flujos", "total_amenazas", "estado", "interfaz"]].copy()
                df_display["fecha_inicio"] = df_display["fecha_inicio"].apply(formato_fecha)
                df_display["fecha_fin"] = df_display["fecha_fin"].apply(formato_fecha)
                df_display.columns = ["ID", "Inicio", "Fin", "Flujos", "Amenazas", "Estado", "Interfaz"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Selector de sesión
                ids_sesiones = [s["id"] for s in sesiones]
                sesion_sel = st.selectbox(
                    "Seleccionar sesion:",
                    ids_sesiones,
                    format_func=lambda sid: next(
                        (f"Sesion {s['id']} - {formato_fecha(s['fecha_inicio'])} ({s['estado']})"
                         for s in sesiones if s['id'] == sid), str(sid)
                    )
                )

                # Detalle de la sesión seleccionada en tabs
                try:
                    resp_detalle = requests.get(
                        f"{BASE_URL}/api/v1/sesiones/{sesion_sel}", timeout=10
                    )
                    if resp_detalle.status_code == 200:
                        detalle = resp_detalle.json()
                        sesion_info = detalle["sesion"]

                        st.subheader(f"Sesion {sesion_info['id']} — {formato_fecha(sesion_info['fecha_inicio'])}")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Flujos totales", sesion_info["total_flujos"])
                        m2.metric("Amenazas", sesion_info["total_amenazas"])
                        m3.metric("Interfaz", sesion_info["interfaz"] or "-")
                        m4.metric("Estado", sesion_info["estado"])

                        tab_dist, tab_temporal, tab_ips, tab_datos, tab_export = st.tabs([
                            "Distribucion", "Detalle temporal", "Trafico por IP", "Datos", "Exportar"
                        ])

                        def colorear_amenazas(row, col="Prediccion"):
                            if row[col] != "BENIGN":
                                return ["color: red; font-weight: bold"] * len(row)
                            return [""] * len(row)

                        # Tab 1: Distribución
                        with tab_dist:
                            distribucion = detalle.get("distribucion", {})
                            if distribucion:
                                col_tabla, col_grafico = st.columns(2)
                                with col_tabla:
                                    df_dist = pd.DataFrame([
                                        {"Prediccion": k, "Total": v,
                                         "Porcentaje": f"{v / sesion_info['total_flujos'] * 100:.1f}%"}
                                        for k, v in sorted(distribucion.items(), key=lambda x: x[1], reverse=True)
                                    ])
                                    st.dataframe(
                                        df_dist.style.apply(colorear_amenazas, axis=1),
                                        use_container_width=True, hide_index=True
                                    )
                                with col_grafico:
                                    fig = px.pie(
                                        names=list(distribucion.keys()),
                                        values=list(distribucion.values()),
                                        title="Distribucion de predicciones"
                                    )
                                    fig.update_layout(height=350)
                                    st.plotly_chart(fig, use_container_width=True)

                        # Tab 2: Detalle temporal
                        with tab_temporal:
                            detalle_pred = detalle.get("detalle_predicciones", [])
                            if detalle_pred:
                                df_detalle = pd.DataFrame(detalle_pred)
                                df_detalle.columns = ["Prediccion", "Total", "Primera deteccion", "Ultima deteccion"]
                                df_detalle["Primera deteccion"] = df_detalle["Primera deteccion"].apply(formato_fecha)
                                df_detalle["Ultima deteccion"] = df_detalle["Ultima deteccion"].apply(formato_fecha)
                                st.dataframe(
                                    df_detalle.style.apply(colorear_amenazas, axis=1),
                                    use_container_width=True, hide_index=True
                                )

                        # Tab 3: Tráfico legítimo vs ataques
                        with tab_ips:
                            distribucion = detalle.get("distribucion", {})
                            if distribucion:
                                total = sesion_info["total_flujos"]
                                benign = distribucion.get("BENIGN", 0)
                                ataques = {k: v for k, v in distribucion.items() if k != "BENIGN"}
                                total_ataques = sum(ataques.values())

                                # Métricas resumen
                                c1, c2 = st.columns(2)
                                c1.metric("Tráfico legítimo (BENIGN)", benign,
                                          f"{benign / total * 100:.1f}% del total" if total else "")
                                c2.metric("Tráfico malicioso (ataques)", total_ataques,
                                          f"{total_ataques / total * 100:.1f}% del total" if total else "")

                                st.markdown("---")

                                # Gráfica de barras: BENIGN vs cada tipo de ataque
                                filas = [{"Tipo": "BENIGN", "Flujos": benign, "Categoria": "Legítimo"}]
                                for tipo, cnt in sorted(ataques.items(), key=lambda x: x[1], reverse=True):
                                    filas.append({"Tipo": tipo, "Flujos": cnt, "Categoria": "Ataque"})
                                df_comp = pd.DataFrame(filas)

                                fig_comp = px.bar(
                                    df_comp,
                                    x="Tipo",
                                    y="Flujos",
                                    color="Categoria",
                                    title="Distribución de flujos: tráfico legítimo vs ataques detectados",
                                    color_discrete_map={"Legítimo": "#2ecc71", "Ataque": "#e74c3c"},
                                    text="Flujos",
                                )
                                fig_comp.update_traces(textposition="outside")
                                fig_comp.update_layout(
                                    xaxis_title="",
                                    legend_title="",
                                    height=420,
                                    showlegend=True,
                                )
                                st.plotly_chart(fig_comp, use_container_width=True)
                            else:
                                st.info("No hay datos para esta sesión")

                        # Tab 4: Datos CSV
                        with tab_datos:
                            resp_csv = requests.get(
                                f"{BASE_URL}/api/v1/export?format=json&sesion_id={sesion_sel}",
                                timeout=30
                            )
                            if resp_csv.status_code == 200:
                                csv_data = resp_csv.json()
                                if csv_data.get("data"):
                                    df_csv = pd.DataFrame(csv_data["data"])

                                    def colorear_filas_csv(row):
                                        pred = row.get("prediccion", "BENIGN")
                                        if pred != "BENIGN":
                                            return ["color: red; font-weight: bold"] * len(row)
                                        return [""] * len(row)

                                    st.dataframe(
                                        df_csv.style.apply(colorear_filas_csv, axis=1),
                                        use_container_width=True, hide_index=True,
                                        height=400
                                    )
                                else:
                                    st.info("No hay datos para esta sesion")

                        # Tab 4: Exportar
                        with tab_export:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Exportar CSV", use_container_width=True):
                                    resp = requests.get(
                                        f"{BASE_URL}/api/v1/export?format=csv&sesion_id={sesion_sel}",
                                        timeout=30
                                    )
                                    if resp.status_code == 200:
                                        st.download_button(
                                            "Descargar CSV",
                                            resp.content,
                                            file_name=f"sesion_{sesion_sel}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.warning("No hay datos para esta sesion")
                            with col2:
                                if st.button("Exportar JSON", use_container_width=True):
                                    resp = requests.get(
                                        f"{BASE_URL}/api/v1/export?format=json&sesion_id={sesion_sel}",
                                        timeout=30
                                    )
                                    if resp.status_code == 200:
                                        st.json(resp.json())
                                    else:
                                        st.warning("No hay datos para esta sesion")

                except Exception as e:
                    st.warning(f"No se pudo cargar el detalle: {e}")
            else:
                st.info("No hay sesiones registradas")
        else:
            st.error("Error al obtener sesiones")
    except Exception as e:
        st.error(f"Error: {e}")


def main():
    """Función principal con navegación"""

    # Sidebar para navegación
    with st.sidebar:
        st.title("️Sistema de Detección de intrusiones")
        st.markdown("---")

        # Menú de navegación
        pagina = st.radio(
            "Navegación:",
            ["Monitoreo en Tiempo Real", "Análisis de PCAP", "Historial de Sesiones"],
            index=0
        )

        st.markdown("---")

        # Selector de interfaz de red
        st.subheader("Interfaz de red")
        iface_data = get_interfaces()
        if iface_data:
            interfaces = iface_data.get("interfaces", [])
            current = iface_data.get("current", "")
            iface_names = [iface["name"] for iface in interfaces]

            # Determinar indice de la interfaz actual
            default_idx = iface_names.index(current) if current in iface_names else 0

            def format_iface(name):
                iface = next((i for i in interfaces if i["name"] == name), {})
                state = iface.get("state", "?")
                ip = iface.get("ip")
                if ip:
                    return f"{name} - {ip} ({state})"
                return f"{name} ({state})"

            selected = st.selectbox(
                "Selecciona interfaz:",
                iface_names,
                index=default_idx,
                format_func=format_iface
            )

            # Mostrar IP de la interfaz actual
            current_iface = next((i for i in interfaces if i["name"] == current), {})
            current_ip = current_iface.get("ip", "Sin IP")
            st.caption(f"IP actual: {current_ip}")

            if selected != current:
                if st.button("Aplicar cambio", type="primary", use_container_width=True):
                    result = cambiar_interface(selected)
                    if result and result.get("status") == "ok":
                        st.success(f"Interfaz cambiada a {selected}")
                        st.rerun()
                    else:
                        st.error("Error al cambiar interfaz")
        else:
            st.warning("No se pudieron obtener las interfaces")

        st.markdown("---")

        # Información de estado
        health = get_api_health()
        if health:
            if health.get('status') == 'ok':
                st.success("API Online")
            else:
                st.error("API Offline")
        else:
            st.warning("API no disponible")

    # Renderizar la página seleccionada
    if pagina == "Monitoreo en Tiempo Real":
        pagina_monitoreo()
    elif pagina == "Análisis de PCAP":
        pagina_analisis_csv()
    elif pagina == "Historial de Sesiones":
        pagina_historial_sesiones()
    
if __name__ == "__main__":
    main()
