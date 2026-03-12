import io
from pathlib import Path
import hmac

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Dashboard Electoral Senado",
    page_icon="🗳️",
    layout="wide",
)

DEFAULT_FILE = Path("john_amaya_senado_20260309_235536.csv")


def check_password():
    if "auth" not in st.secrets:
        st.error("No se han configurado las credenciales de acceso en Streamlit Secrets.")
        st.info("Ve a Manage app → Settings → Secrets y agrega el bloque [auth].")
        return False

    if "username" not in st.secrets["auth"] or "password" not in st.secrets["auth"]:
        st.error("Faltan username o password dentro de [auth] en Streamlit Secrets.")
        st.info("Ejemplo válido:\n\n[auth]\nusername = \"admin\"\npassword = \"ClaveSegura123\"")
        return False

    def password_entered():
        usuario_ok = hmac.compare_digest(
            st.session_state.get("username", ""),
            st.secrets["auth"]["username"],
        )

        password_ok = hmac.compare_digest(
            st.session_state.get("password", ""),
            st.secrets["auth"]["password"],
        )

        if usuario_ok and password_ok:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("Acceso al Dashboard Electoral")
    st.text_input("Usuario", key="username")
    st.text_input("Contraseña", type="password", key="password", on_change=password_entered)

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Usuario o contraseña incorrectos")

    return False


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if DEFAULT_FILE.exists():
        return pd.read_csv(DEFAULT_FILE)

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    numeric_cols = [
        "amb_depto",
        "amb_municipio",
        "total_mesas",
        "mesas_escrutadas",
        "pct_mesas",
        "potencial_electoral",
        "votantes_municipio",
        "pct_participacion",
        "votos_validos",
        "votos_blancos",
        "codpar",
        "votos_partido_muni",
        "cedula",
        "votos_candidato",
        "pct_sobre_partido",
        "voto_preferente",
        "electo",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["depto", "municipio", "candidato"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    return out


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"votantes_municipio", "potencial_electoral"}.issubset(out.columns):
        out["participacion_calc"] = np.where(
            out["potencial_electoral"] > 0,
            out["votantes_municipio"] / out["potencial_electoral"],
            np.nan,
        )
    else:
        out["participacion_calc"] = np.nan

    if {"votos_candidato", "votantes_municipio"}.issubset(out.columns):
        out["fuerza_candidato"] = np.where(
            out["votantes_municipio"] > 0,
            out["votos_candidato"] / out["votantes_municipio"],
            np.nan,
        )
    else:
        out["fuerza_candidato"] = np.nan

    if {"votos_candidato", "votos_partido_muni"}.issubset(out.columns):
        out["dependencia_partido"] = np.where(
            out["votos_partido_muni"] > 0,
            out["votos_candidato"] / out["votos_partido_muni"],
            np.nan,
        )
    else:
        out["dependencia_partido"] = np.nan

    if {"votos_candidato", "potencial_electoral"}.issubset(out.columns):
        out["eficiencia_electoral"] = np.where(
            out["potencial_electoral"] > 0,
            out["votos_candidato"] / out["potencial_electoral"],
            np.nan,
        )
    else:
        out["eficiencia_electoral"] = np.nan

    return out


def get_optional_geo_levels(df: pd.DataFrame):
    optional_levels = []
    for candidate in ["puesto", "puesto_votacion", "nombre_puesto", "mesa"]:
        if candidate in df.columns:
            optional_levels.append(candidate)
    return optional_levels


def apply_filters(df: pd.DataFrame):
    st.sidebar.header("Filtros")

    candidatos = sorted(df["candidato"].dropna().unique().tolist()) if "candidato" in df.columns else []
    deptos = sorted(df["depto"].dropna().unique().tolist()) if "depto" in df.columns else []

    selected_candidatos = st.sidebar.multiselect(
        "Candidato",
        options=candidatos,
        default=candidatos[:1] if len(candidatos) == 1 else candidatos,
    )

    selected_deptos = st.sidebar.multiselect(
        "Departamento",
        options=deptos,
        default=deptos,
    )

    filtered = df.copy()
    if selected_candidatos:
        filtered = filtered[filtered["candidato"].isin(selected_candidatos)]
    if selected_deptos:
        filtered = filtered[filtered["depto"].isin(selected_deptos)]

    municipios = sorted(filtered["municipio"].dropna().unique().tolist()) if "municipio" in filtered.columns else []
    selected_municipios = st.sidebar.multiselect(
        "Municipio",
        options=municipios,
        default=municipios,
    )
    if selected_municipios:
        filtered = filtered[filtered["municipio"].isin(selected_municipios)]

    for optional_col in get_optional_geo_levels(filtered):
        values = sorted(filtered[optional_col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(
            optional_col.replace("_", " ").title(),
            options=values,
            default=values,
        )
        if selected:
            filtered = filtered[filtered[optional_col].astype(str).isin(selected)]

    return filtered


def format_int(value):
    if pd.isna(value):
        return "—"
    return f"{int(round(value)):,}".replace(",", ".")


def format_pct(value):
    if pd.isna(value):
        return "—"
    return f"{value:.2%}"


def build_summary_metrics(df: pd.DataFrame):
    total_votos_candidato = df["votos_candidato"].sum() if "votos_candidato" in df.columns else np.nan
    total_votos_partido = df["votos_partido_muni"].sum() if "votos_partido_muni" in df.columns else np.nan
    total_potencial = df["potencial_electoral"].sum() if "potencial_electoral" in df.columns else np.nan
    total_votantes = df["votantes_municipio"].sum() if "votantes_municipio" in df.columns else np.nan

    participacion = total_votantes / total_potencial if total_potencial and total_potencial > 0 else np.nan
    dependencia = total_votos_candidato / total_votos_partido if total_votos_partido and total_votos_partido > 0 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Votos candidato", format_int(total_votos_candidato))
    c2.metric("Votos partido", format_int(total_votos_partido))
    c3.metric("Potencial electoral", format_int(total_potencial))
    c4.metric("Participación", format_pct(participacion))
    c5.metric("Dependencia del partido", format_pct(dependencia if not pd.isna(dependencia) else np.nan))

    st.caption(
        "La participación se calcula como votantes / potencial electoral. "
        "La dependencia del partido se calcula como votos del candidato / votos del partido."
    )


def aggregate_territory(df: pd.DataFrame, level: str) -> pd.DataFrame:
    agg = (
        df.groupby(level, dropna=False, as_index=False)
        .agg(
            potencial_electoral=("potencial_electoral", "sum"),
            votantes_municipio=("votantes_municipio", "sum"),
            votos_validos=("votos_validos", "sum"),
            votos_blancos=("votos_blancos", "sum"),
            votos_partido_muni=("votos_partido_muni", "sum"),
            votos_candidato=("votos_candidato", "sum"),
            total_mesas=("total_mesas", "sum"),
            mesas_escrutadas=("mesas_escrutadas", "sum"),
        )
    )

    agg["participacion"] = np.where(
        agg["potencial_electoral"] > 0,
        agg["votantes_municipio"] / agg["potencial_electoral"],
        np.nan,
    )
    agg["fuerza_candidato"] = np.where(
        agg["votantes_municipio"] > 0,
        agg["votos_candidato"] / agg["votantes_municipio"],
        np.nan,
    )
    agg["dependencia_partido"] = np.where(
        agg["votos_partido_muni"] > 0,
        agg["votos_candidato"] / agg["votos_partido_muni"],
        np.nan,
    )

    total_candidato = agg["votos_candidato"].sum()
    agg["peso_territorial"] = np.where(
        total_candidato > 0,
        agg["votos_candidato"] / total_candidato,
        np.nan,
    )
    agg["eficiencia_electoral"] = np.where(
        agg["potencial_electoral"] > 0,
        agg["votos_candidato"] / agg["potencial_electoral"],
        np.nan,
    )

    conditions = [
        (agg["votos_candidato"] >= agg["votos_candidato"].quantile(0.75))
        & (agg["participacion"] >= agg["participacion"].quantile(0.75)),
        (agg["potencial_electoral"] >= agg["potencial_electoral"].quantile(0.75))
        & (agg["votos_candidato"] <= agg["votos_candidato"].quantile(0.25)),
        (agg["potencial_electoral"] >= agg["potencial_electoral"].quantile(0.75))
        & (agg["participacion"] <= agg["participacion"].quantile(0.25)),
    ]
    choices = ["FORTALEZA", "OPORTUNIDAD", "RIESGO"]
    agg["segmento_estrategico"] = np.select(conditions, choices, default="ESTABLE")

    return agg.sort_values("votos_candidato", ascending=False)


def render_charts(df: pd.DataFrame):
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Territorio", "Participación", "Partido vs candidato", "Insights y tabla"]
    )

    level = "municipio" if "municipio" in df.columns else "depto"
    territory = aggregate_territory(df, level)

    with tab1:
        c1, c2 = st.columns(2)

        fig_bar = px.bar(
            territory.head(20),
            x=level,
            y="votos_candidato",
            hover_data=["votos_partido_muni", "participacion", "peso_territorial"],
            title=f"Top 20 por votos del candidato ({level.title()})",
        )
        fig_bar.update_layout(xaxis_title=level.title(), yaxis_title="Votos candidato")
        c1.plotly_chart(fig_bar, use_container_width=True)

        fig_treemap = px.treemap(
            territory.head(50),
            path=[level],
            values="votos_candidato",
            color="participacion",
            hover_data=["potencial_electoral", "votantes_municipio", "dependencia_partido"],
            title=f"Peso territorial y participación por {level}",
        )
        c2.plotly_chart(fig_treemap, use_container_width=True)

        depto_level = aggregate_territory(df, "depto") if "depto" in df.columns else territory
        fig_depto = px.bar(
            depto_level.sort_values("votos_candidato", ascending=False),
            x="depto",
            y="votos_candidato",
            color="segmento_estrategico",
            hover_data=["participacion", "potencial_electoral", "dependencia_partido"],
            title="Rendimiento por departamento",
        )
        st.plotly_chart(fig_depto, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        fig_part = px.scatter(
            territory,
            x="potencial_electoral",
            y="participacion",
            size="votos_candidato",
            color="segmento_estrategico",
            hover_name=level,
            title=f"Participación vs potencial electoral por {level}",
        )
        c1.plotly_chart(fig_part, use_container_width=True)

        territory["abstencion_estimada"] = territory["potencial_electoral"] - territory["votantes_municipio"]
        fig_abs = px.bar(
            territory.sort_values("abstencion_estimada", ascending=False).head(20),
            x=level,
            y="abstencion_estimada",
            title=f"Top 20 con mayor abstención estimada por {level}",
        )
        c2.plotly_chart(fig_abs, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)

        fig_rel = px.scatter(
            territory,
            x="votos_partido_muni",
            y="votos_candidato",
            color="segmento_estrategico",
            size="potencial_electoral",
            hover_name=level,
            title="Relación votos partido vs votos candidato",
        )
        c1.plotly_chart(fig_rel, use_container_width=True)

        fig_dep = px.bar(
            territory.sort_values("dependencia_partido", ascending=False).head(20),
            x=level,
            y="dependencia_partido",
            color="segmento_estrategico",
            title=f"Top 20 por dependencia del partido ({level})",
        )
        c2.plotly_chart(fig_dep, use_container_width=True)

    with tab4:
        fuertes = territory.sort_values("votos_candidato", ascending=False).head(5)
        oportunidades = territory[territory["segmento_estrategico"] == "OPORTUNIDAD"].sort_values(
            "potencial_electoral", ascending=False
        ).head(5)
        riesgos = territory[territory["segmento_estrategico"] == "RIESGO"].sort_values(
            "potencial_electoral", ascending=False
        ).head(5)

        st.subheader("Insights automáticos")

        if not fuertes.empty:
            st.markdown(
                "**Territorios más fuertes:** "
                + ", ".join(
                    [
                        f"{row[level]} ({format_int(row['votos_candidato'])} votos; peso {format_pct(row['peso_territorial'])})"
                        for _, row in fuertes.iterrows()
                    ]
                )
            )

        if not oportunidades.empty:
            st.markdown(
                "**Territorios de oportunidad:** "
                + ", ".join(
                    [
                        f"{row[level]} (potencial {format_int(row['potencial_electoral'])}, votos {format_int(row['votos_candidato'])})"
                        for _, row in oportunidades.iterrows()
                    ]
                )
            )

        if not riesgos.empty:
            st.markdown(
                "**Territorios en riesgo:** "
                + ", ".join(
                    [
                        f"{row[level]} (participación {format_pct(row['participacion'])}, potencial {format_int(row['potencial_electoral'])})"
                        for _, row in riesgos.iterrows()
                    ]
                )
            )

        st.subheader("Tabla analítica")
        st.dataframe(
            territory,
            use_container_width=True,
            hide_index=True,
        )

        csv = territory.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar tabla analítica",
            data=csv,
            file_name="tabla_analitica_electoral.csv",
            mime="text/csv",
        )


def main():
    if not check_password():
        st.stop()

    st.title("🗳️ Dashboard interactivo de resultados electorales")
    st.write(
        "Este dashboard analiza resultados de Senado por territorio y genera insights tácticos. "
        "Funciona con el archivo entregado y también permite cargar un CSV propio."
    )

    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    df = load_data(uploaded_file)

    if df is None:
        st.warning("No se encontró un archivo por defecto. Carga un CSV para iniciar.")
        st.stop()

    df = normalize_columns(df)
    df = add_metrics(df)

    missing = [c for c in ["depto", "municipio", "candidato", "votos_candidato"] if c not in df.columns]
    if missing:
        st.error(f"El archivo no contiene columnas clave requeridas: {', '.join(missing)}")
        st.stop()

    if "mesa" not in df.columns and "puesto" not in df.columns and "puesto_votacion" not in df.columns:
        st.info(
            "El archivo actual llega hasta nivel municipio. "
            "El dashboard ya está preparado para drill-down adicional si en el futuro agregas columnas de puesto y/o mesa."
        )

    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("Los filtros seleccionados no tienen datos.")
        st.stop()

    build_summary_metrics(filtered)
    render_charts(filtered)

    with st.expander("Ver estructura y calidad de datos"):
        st.write("Filas:", len(filtered))
        st.write("Columnas:", list(filtered.columns))
        st.dataframe(filtered.head(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
