import random
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from utils.api_client import PumpAPIClient


st.set_page_config(page_title="Pump Failure Monitoring", page_icon="🔄", layout="wide")


@st.cache_resource
def get_api_client():
    return PumpAPIClient()


api_client = get_api_client()


if "history" not in st.session_state:
    st.session_state.history = []


def calculate_combined_features(temp, pressure, vibration, rpm, flow_rate):
    temp_pressure = temp / (pressure + 1e-10)
    flow_temp = flow_rate / (temp + 1e-10)
    vibration_rpm = vibration / (rpm + 1e-10)
    return temp_pressure, flow_temp, vibration_rpm


with st.sidebar:
    st.header("⚙️ Настройки")
    pump_id = st.selectbox("Насос", [1, 2, 3, 4, 5], index=0)
    st.markdown("---")

    mode = st.radio("Режим ввода", ["📌 Ручной", "🖧 Авто-режим (имитация ПЛК)"])

    if mode == "🖧 Авто-режим (имитация ПЛК)":
        auto_pump = st.selectbox("Насос для авто-режима", [1, 2, 3, 4, 5], index=1)
        interval = st.slider("Интервал обновления", 2, 10, 5)
        if st.button("▶️ Запустить авто-режим", type="primary", use_container_width=True):
            st.session_state.auto_running = True
        if st.button("⏹️ Остановить", use_container_width=True):
            st.session_state.auto_running = False

    st.markdown("---")

    if api_client.health_check():
        st.success("✅ API сервис доступен")
    else:
        st.error("❌ API сервис недоступен")

    st.markdown("---")
    st.caption("Цветовая индикация:")
    st.markdown("🟢 **0-40%** — Норма")
    st.markdown("🟡 **40-70%** — Внимание")
    st.markdown("🔴 **70-100%** — Опасно")

st.title("🔄 Система мониторинга состояния насосного оборудования")
st.markdown("---")

if mode == "📌 Ручной":
    st.header("📝 Ввод показаний датчиков")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input("🌡️ Температура (°C)", value=95.0, step=0.5, format="%.1f")
        vibration = st.number_input("📳 Вибрация (мм/с)", value=2.5, step=0.1, format="%.1f")

    with col2:
        pressure = st.number_input("⏲️ Давление (бар)", value=180.0, step=1.0, format="%.1f")
        flow_rate = st.number_input("💧 Расход (м³/ч)", value=8.0, step=0.1, format="%.1f")

    with col3:
        rpm = st.number_input("🔄 Обороты (об/мин)", value=1850, step=2)
        operational_hours = st.number_input("⏱️ Часы работы", value=12000, step=1)

    predict_button = st.button("Получить прогноз", type="primary", use_container_width=True)

    if predict_button:
        data = {
            "pump_id": pump_id,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "flow_rate": flow_rate,
            "rpm": rpm,
            "operational_hours": operational_hours,
        }

        with st.spinner("Идет прогноз"):
            result = api_client.predict(data)

        if result and "error" not in result:
            st.session_state.history.append(
                {
                    "timestamp": datetime.now(),
                    "pump_id": pump_id,
                    "temperature": temperature,
                    "vibration": vibration,
                    "pressure": pressure,
                    "flow_rate": flow_rate,
                    "rpm": rpm,
                    "probability": result["probability"],
                    "risk_level": result["risk_level"],
                    "status": result["status"],
                }
            )
            if len(st.session_state.history) > 100:
                st.session_state.history = st.session_state.history[-100:]

            st.success("Прогноз получен")
        else:
            st.error("Ошибка, проверьте подключения")

if mode == "🖧 Авто-режим (имитация ПЛК)" and st.session_state.get("auto_running", False):
    placeholder = st.empty()

    pump_configs = {
        1: {"temp": 95.2, "vib": 2.1, "press": 190.5, "flow": 8.1, "rpm": 1850},
        2: {"temp": 112.5, "vib": 3.9, "press": 155.2, "flow": 5.8, "rpm": 1845},
        3: {"temp": 102.3, "vib": 3.2, "press": 168.7, "flow": 6.9, "rpm": 1860},
        4: {"temp": 97.8, "vib": 2.4, "press": 185.3, "flow": 7.8, "rpm": 1855},
        5: {"temp": 94.5, "vib": 2.0, "press": 192.1, "flow": 8.3, "rpm": 1840},
    }

    info = placeholder.info("🔄 Авто-режим запущен...")

    for i in range(50):
        if not st.session_state.get("auto_running", False):
            break

        config = pump_configs[auto_pump]
        data = {
            "pump_id": auto_pump,
            "temperature": config["temp"] + random.uniform(-1.5, 1.5),
            "vibration": config["vib"] + random.uniform(-0.15, 0.15),
            "pressure": config["press"] + random.uniform(-3, 3),
            "flow_rate": config["flow"] + random.uniform(-0.2, 0.2),
            "rpm": config["rpm"] + random.uniform(-10, 10),
            "operational_hours": 12000 + i * 5,
        }

        result = api_client.predict(data)

        if result and "error" not in result:
            st.session_state.history.append(
                {
                    "timestamp": datetime.now(),
                    "pump_id": auto_pump,
                    **data,
                    "probability": result["probability"],
                    "risk_level": result["risk_level"],
                    "status": result["status"],
                }
            )

            if len(st.session_state.history) > 50:
                st.session_state.history = st.session_state.history[-50:]

            info.info(
                f"🔄 Насос {auto_pump}: вероятность {result['probability']*100:.1f}% "
                f"({result['status']})"
            )

        time.sleep(interval)

    placeholder.empty()
    st.session_state.auto_running = False
    st.success("Демонстрация завершена")


if st.session_state.history:
    last = st.session_state.history[-1]

    st.markdown("---")
    st.header("Текущий прогноз")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        prob = last["probability"]

        if prob < 0.4:
            color, level, bg = "green", "Нормальный режим", "#0b4c0b"
        elif prob < 0.7:
            color, level, bg = "orange", "Обратите внимание", "#c3bb78"
        else:
            color, level, bg = "red", "ОПАСНО", "#d28a95"

        st.markdown(
            f"""
        <div style="background-color:{bg}; padding:20px; border-radius:15px; text-align:center;">
            <h2 style="color:black;">{level}</h2>
            <p style="font-size:48px; font-weight:bold; margin:0;">{prob*100:.1f}%</p>
            <p>вероятность отказа оборудования</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Температура", f"{last['temperature']:.1f}°C")
        st.metric("Давление", f"{last['pressure']:.0f} бар")
        st.metric("Вибрация", f"{last['vibration']:.1f} мм/с")

    with col3:
        st.metric("Расход", f"{last['flow_rate']:.1f} м³/ч")
        st.metric("Обороты", f"{last['rpm']:.0f} об/мин")
        st.metric("Насос", f"№{last['pump_id']}")


if len(st.session_state.history) >= 2:
    st.markdown("---")
    st.subheader("📈 Тренд вероятности отказа (история)")

    df_history = pd.DataFrame(st.session_state.history)

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=df_history["timestamp"],
            y=df_history["probability"],
            mode="lines+markers",
            name="Вероятность отказа",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )

    fig1.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Порог (0.5)")
    fig1.add_hrect(y0=0, y1=0.3, line_width=0, fillcolor="green", opacity=0.1)
    fig1.add_hrect(y0=0.3, y1=0.7, line_width=0, fillcolor="yellow", opacity=0.1)
    fig1.add_hrect(y0=0.7, y1=1, line_width=0, fillcolor="red", opacity=0.1)

    fig1.update_layout(
        xaxis_title="Время", yaxis_title="Вероятность отказа", height=350, showlegend=True
    )
    fig1.update_yaxes(range=[0, 1])

    st.plotly_chart(fig1, use_container_width=True)

if len(st.session_state.history) >= 2:
    st.subheader("📊 Динамика комбинаторных признаков")

    df_combined = df_history.copy()
    combined = df_combined.apply(
        lambda row: calculate_combined_features(
            row["temperature"], row["pressure"], row["vibration"], row["rpm"], row["flow_rate"]
        ),
        axis=1,
    )
    df_combined["temp_pressure"] = [c[0] for c in combined]
    df_combined["flow_temp"] = [c[1] for c in combined]
    df_combined["vibration_rpm"] = [c[2] for c in combined]

    fig2 = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "temp_pressure (температура/давление)",
            "flow_temp (расход/температура)",
            "vibration_rpm (вибрация/обороты)",
        ),
    )

    fig2.add_trace(
        go.Scatter(
            x=df_combined["timestamp"],
            y=df_combined["temp_pressure"],
            mode="lines+markers",
            name="temp_pressure",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=df_combined["timestamp"],
            y=df_combined["flow_temp"],
            mode="lines+markers",
            name="flow_temp",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=df_combined["timestamp"],
            y=df_combined["vibration_rpm"],
            mode="lines+markers",
            name="vibration_rpm",
            line=dict(color="purple"),
        ),
        row=3,
        col=1,
    )

    fig2.update_layout(height=600, showlegend=True)
    fig2.update_xaxes(title_text="Время", row=3, col=1)

    st.plotly_chart(fig2, use_container_width=True)

    st.info("Подключите систему или введите данные в ручную.")

st.markdown("---")
st.caption("🟢 Зеленый: низкий риск | 🟡 Желтый: средний риск | 🔴 Красный: высокий риск")
