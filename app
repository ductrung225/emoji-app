import streamlit as st
import numpy as np
import cv2

from keras.models import load_model
from streamlit_drawable_canvas import st_canvas


model = load_model("emoji_ann_model.h5")

emoji_names = [
    "😀 Vui",
    "😢 Buồn",
    "😡 Tức giận",
    "😭 Khóc",
    "😲 Ngạc nhiên",
    "🖐 Bàn tay",
    "🇻🇳 Cờ Việt Nam",
    "🕒 Đồng hồ",
    "☁️ Mây",
    "🔥 Lửa"
]

st.title("Nhận dạng 10 Emoji bằng ANN")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=8,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Dự đoán"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        img = cv2.cvtColor(
            img.astype('uint8'),
            cv2.COLOR_RGBA2GRAY
        )

        img = cv2.resize(img, (50, 50))

        img = img.reshape(1, 2500)

        img = img.astype('float32') / 255

        pred = model.predict(img)

        label = np.argmax(pred)

        confidence = np.max(pred)

        st.subheader(
            f"Kết quả: {emoji_names[label]}"
        )

        st.write(
            f"Độ tin cậy: {confidence*100:.2f}%"
        )
