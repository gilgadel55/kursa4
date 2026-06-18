import streamlit as st
import numpy as np
import pandas as pd
import tifffile
import os
from skimage import filters, measure, morphology
import plotly.graph_objects as go


# =============================================================================
# 🔧 МОДУЛИ ОБРАБОТКИ
# =============================================================================

def load_tomography(file_obj):
    """Загрузка 3D-томограммы из .npy или .tiff"""
    ext = file_obj.name.split('.')[-1].lower()
    if ext == 'npy':
        return np.load(file_obj)
    elif ext in ('tif', 'tiff'):
        return tifffile.imread(file_obj)
    else:
        raise ValueError("Формат не поддерживается. Используйте .npy или .tiff")


def preprocess_and_segment(arr, sigma=1.0, min_size=50):
    """Предобработка: сглаживание → Otsu → морфология"""
    arr_smooth = filters.gaussian(arr, sigma=sigma)
    thresh = filters.threshold_otsu(arr_smooth)
    mask = arr_smooth > thresh
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.binary_closing(mask, morphology.ball(2))
    return arr_smooth, mask


def compute_metrics(mask, voxel_size_mm):
    """Расчёт фенотипических признаков"""
    metrics = {"Объём (мм³)": 0.0, "Площадь пов. (мм²)": 0.0, "Длина (мм)": 0.0}
    if not np.any(mask):
        return pd.DataFrame([metrics])

    # Объём
    metrics["Объём (мм³)"] = float(np.sum(mask) * (voxel_size_mm ** 3))

    # 3D-сетка для площади и длины
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    metrics["Площадь пов. (мм²)"] = float(measure.mesh_surface_area(verts, faces) * (voxel_size_mm ** 2))

    # Длина по оси Z (высота растения/корня)
    z_coords = np.argwhere(mask)[:, 0]
    metrics["Длина (мм)"] = float((z_coords.max() - z_coords.min() + 1) * voxel_size_mm)

    return pd.DataFrame([metrics])


def reconstruct_mesh(mask):
    """Построение 3D-поверхности (Marching Cubes)"""
    if not np.any(mask):
        return None, None
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return verts, faces


def generate_demo_data():
    """Синтетическая томограмма для мгновенного демо"""
    shape = (64, 64, 64)
    vol = np.zeros(shape, dtype=np.float32)
    y, x, z = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # Стебель (цилиндр)
    stem = (x - 32) ** 2 + (y - 32) ** 2 < 5 ** 2
    vol[stem & (z > 10) & (z < 55)] = 160

    # Листья/ветви (сферы)
    centers = [(32, 12, 25), (32, 52, 35), (12, 32, 45), (52, 32, 20)]
    for cx, cy, cz in centers:
        leaf = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < 9 ** 2
        vol[leaf] = 130

    # Шум и нормализация
    vol += np.random.normal(0, 12, shape)
    vol = np.clip(vol, 0, 255).astype(np.float32)
    return vol


def save_obj(verts, faces, filepath):
    """Экспорт сетки в формат OBJ"""
    with open(filepath, 'w') as f:
        f.write("# Plant 3D Model\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


# =============================================================================
# 🖥️ ИНТЕРФЕЙС STREAMLIT
# =============================================================================

def main():
    st.set_page_config(page_title="Цифровое фенотипирование", layout="wide")
    st.title("🌿 ИС анализа томографических изображений растений")
    st.caption("Прототип MVP: загрузка → сегментация → 3D-реконструкция → извлечение метрик")

    # Боковая панель параметров
    with st.sidebar:
        st.header("⚙️ Параметры обработки")
        voxel_size = st.number_input("Размер вокселя (мм)", min_value=0.01, max_value=5.0, value=0.1, step=0.01,
                                     help="Физический размер одного вокселя из метаданных томографа")
        sigma = st.slider("Радиус сглаживания (σ)", 0.5, 3.0, 1.0, 0.1)
        min_obj_size = st.number_input("Мин. размер объекта (вокселей)", 10, 500, 50)
        demo_mode = st.checkbox("🎯 Использовать демо-данные", value=False)

    # Загрузка данных
    uploaded_file = None
    if not demo_mode:
        uploaded_file = st.file_uploader("Загрузите файл .npy или .tiff", type=["npy", "tif", "tiff"])

    if uploaded_file or demo_mode:
        if st.button("🚀 Запустить анализ", type="primary"):
            progress = st.progress(0, text="Инициализация...")
            try:
                # 1. Загрузка
                progress.progress(10, text="Загрузка томограммы...")
                arr = generate_demo_data() if demo_mode else load_tomography(uploaded_file)

                # 2. Сегментация
                progress.progress(35, text="Предобработка и выделение структуры...")
                _, mask = preprocess_and_segment(arr, sigma=sigma, min_size=min_obj_size)

                if not np.any(mask):
                    st.error("⚠️ Объект не обнаружен. Уменьшите мин. размер объекта или радиус сглаживания.")
                    progress.empty()
                    return

                # 3. Метрики
                progress.progress(60, text="Расчёт фенотипических признаков...")
                df_metrics = compute_metrics(mask, voxel_size)

                # 4. 3D-реконструкция
                progress.progress(80, text="Построение 3D-поверхности...")
                verts, faces = reconstruct_mesh(mask)

                # === ВИЗУАЛИЗАЦИЯ И ЭКСПОРТ ===
                st.success("✅ Анализ завершён успешно!")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("📊 Извлечённые признаки")
                    st.dataframe(df_metrics.style.format("{:.3f}"), use_container_width=True)

                    csv_bytes = df_metrics.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Скачать метрики (CSV)", csv_bytes, "phenotype_metrics.csv", "text/csv")

                with col2:
                    st.subheader("🔍 Интерактивная 3D-модель")
                    fig = go.Figure(data=[go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color="#4CAF50", opacity=0.75,
                        lighting=dict(ambient=0.4, diffuse=0.9, specular=0.3)
                    )])
                    fig.update_layout(
                        scene=dict(aspectmode='data', bgcolor='#f0f2f6'),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Экспорт OBJ
                    tmp_obj = "temp_plant.obj"
                    save_obj(verts, faces, tmp_obj)
                    with open(tmp_obj, "rb") as f:
                        st.download_button("💾 Скачать 3D-модель (OBJ)", f, file_name="plant_model.obj",
                                           mime="model/obj")
                    if os.path.exists(tmp_obj):
                        os.remove(tmp_obj)

                progress.progress(100, text="Готово!")

            except Exception as e:
                st.error(f"❌ Ошибка выполнения: {str(e)}")
            finally:
                progress.empty()
    else:
        st.info("👆 Загрузите файл томограммы или включите демо-режим в левой панели.")


if __name__ == "__main__":
    main()