# streamlit_yolov8_eye_detect.py

import os
import subprocess
import sys

# Cek apakah streamlit sudah terinstall, kalau belum maka install
try:
    import streamlit as st
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    import streamlit as st


import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import json
from glob import glob
from collections import Counter
from ultralytics import YOLO
import shutil



# --- Buat page nya yahhh lovv---
st.set_page_config(page_title="Deteksi Penyakit Mata", page_icon="👁️", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #E0F7F1;
        }
    </style>
""", unsafe_allow_html=True)


from PIL import Image

# === SIDEBAR MENU ===
st.sidebar.markdown("## 📋 Menu")
menu = st.sidebar.selectbox("Navigasi", ["Home", "Deteksi"])


if menu == "Home":
    st.title("🧿 Edukasi Penyakit Mata")
    st.markdown("Berikut adalah beberapa jenis penyakit mata yang dapat terdeteksi oleh sistem ini:")

    penyakit_info = {
        "Glaucoma": {
            "deskripsi": "Glaukoma adalah sekelompok penyakit mata yang merusak saraf optik dan dapat menyebabkan kebutaan jika tidak diobati.",
            "gambar": "glaucoma.jpeg"
        },
        "Uveitis": {
            "deskripsi": "Uveitis adalah peradangan pada lapisan tengah mata (uvea), yang bisa menyebabkan kemerahan, nyeri, dan penglihatan kabur.",
            "gambar": "uveitis.jpeg"
        },
        "Cataracts": {
            "deskripsi": "Katarak menyebabkan lensa mata menjadi keruh, biasanya karena penuaan, dan dapat menyebabkan penglihatan kabur.",
            "gambar": "Cataracts.jpeg"
        },
        "Bulging Eyes": {
            "deskripsi": "Bulging Eyes atau exophthalmos adalah kondisi di mana mata tampak menonjol keluar, sering dikaitkan dengan penyakit Graves.",
            "gambar": "Bulging_eyes.jpeg"
        },
        "Crossed Eyes": {
            "deskripsi": "Crossed Eyes atau strabismus adalah kondisi di mana mata tidak sejajar satu sama lain, menyebabkan penglihatan ganda.",
            "gambar": "Crosssed_eyes.jpeg"
        }
    }

    for nama, info in penyakit_info.items():
        st.subheader(nama)
        st.write(info["deskripsi"])
        try:
            img = Image.open(info["gambar"])
            st.image(img, caption=f"Contoh gambar {nama}", width=300)
        except:
            st.warning(f"Gambar untuk {nama} belum tersedia.")

    st.stop()  # Hentikan eksekusi jika di Home


# --- CLASS NAMES (sesuaikan dengan dataset Roboflow kamu) ---
class_names = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Glaucoma', 'Uveitis']

# --- LOAD MODEL YOLOv8 ---
model = YOLO("runs/train/YOLOv8_EYE_DISEASES/weights/best.pt")  # GANTI dari 'yolov8s.pt' ke 'best.pt'

# --- LOAD TREATMENT INFORMATION ---
@st.cache_data
def load_penanganan(path="treatment.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        #st.success("✅ Berhasil memuat treatment.json")
        return {item["label"]: item["instructions"] for item in data}
    except Exception as e:
        st.error(f"❌ Gagal memuat treatment.json: {e}")
        return {}

penanganan_dict = load_penanganan()


# --- HELPER FUNCTIONS ---
def load_image_opencv(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def run_detection_logic(file):
    image = load_image_opencv(file)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    # Hapus hasil lama
    if os.path.exists("runs/detect/predict"):
        shutil.rmtree("runs/detect/predict")

    result = {"filename": file.name, "labels": [], "image_path": None, "error": None}
    try:
        yolo_result = model.predict(
            tmp_path,
            conf=0.5,
            save=True,
            save_txt=True,
            show_labels=True,   # tampilkan label
            show_conf=True,     # tampilkan confidence (satu nilai saja)
            show_boxes=True,    # tampilkan kotak
            verbose=False       # agar tidak muncul log berlebihan
        )
        boxes = yolo_result[0].boxes
        if boxes is not None and len(boxes) > 0:
            scores = boxes.conf.cpu().numpy()  # confidence scores
            classes = boxes.cls.cpu().numpy().astype(int)
            top_index = int(np.argmax(scores))  # ambil index dengan confidence tertinggi
            top_label = class_names[classes[top_index]]
            result["labels"] = [top_label]
        else:
            result["labels"] = []


        latest_img = glob("runs/detect/predict*/" + os.path.basename(tmp_path))
        if latest_img:
            result["image_path"] = latest_img[0]
        else:
            result["error"] = "Gambar hasil deteksi tidak ditemukan."
    except Exception as e:
        result["error"] = f"Deteksi gagal: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return result

# --- MAIN APP ---
st.header("Deteksi Penyakit Mata Menggunakan YOLOv8")

st.markdown("Silakan unggah gambar mata untuk dideteksi apakah mengandung gejala penyakit tertentu.")

if 'detection_results' not in st.session_state:
    st.session_state['detection_results'] = []
if 'uploaded_file_key' not in st.session_state:
    st.session_state['uploaded_file_key'] = 0

uploaded_file = st.file_uploader("Upload Gambar Mata", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state['uploaded_file_key']}")
col1, col2 = st.columns(2)
with col1:
    detect_btn = st.button("Mulai Deteksi", use_container_width=True)
with col2:
    clear_btn = st.button("Hapus Gambar", use_container_width=True)

if detect_btn:
    if uploaded_file is None:
        st.warning("⚠️ Harap upload gambar terlebih dahulu.")
    else:
        with st.spinner("🔍 Mendeteksi penyakit mata..."):
            result = run_detection_logic(uploaded_file)
            st.session_state['detection_results'].append(result)

if clear_btn:
    st.session_state['detection_results'] = []
    st.session_state['uploaded_file_key'] += 1
    st.rerun()

if st.session_state['detection_results']:
    for result in st.session_state['detection_results']:
        st.markdown(f"---\n### Hasil Deteksi untuk: {result['filename']}")

        if result.get("error"):
            st.error(f"Terjadi kesalahan: {result['error']}")

        if result['image_path'] and os.path.exists(result["image_path"]):
            st.image(result["image_path"], caption="Gambar dengan Deteksi", use_container_width=True)
        elif uploaded_file and not result.get("error"):
            st.image(uploaded_file, caption="Gambar Asli", use_container_width=True)

        if result['labels']:
            counts = Counter(result["labels"])
            st.markdown("---\n#### ✅ Penyakit Mata yang Terdeteksi:")
            for label, count in counts.items():
                display_label = label.split()[0].replace("_", " ").title()
                st.success(f"**{display_label}**")

            st.markdown("---\n#### 📝 Saran Penanganan:")
            displayed_treatments = set()
            for label in result["labels"]:
                if label not in displayed_treatments:
                    treatment_steps = penanganan_dict.get(label)
                    display_label = label.replace("_", " ").title()
                    if treatment_steps:
                        with st.expander(f"Penanganan untuk {display_label}", expanded=False):
                            for i, step in enumerate(treatment_steps, 1):
                                st.markdown(f"{i}. {step}")
                    else:
                        st.info(f"Tidak ada informasi penanganan spesifik untuk {display_label}.")
                    displayed_treatments.add(label)
        elif not result.get("error"):
            st.info("Tidak ada penyakit yang terdeteksi pada gambar ini.")

elif not uploaded_file:
    st.info("Silakan upload gambar dan klik 'Mulai Deteksi'.")
