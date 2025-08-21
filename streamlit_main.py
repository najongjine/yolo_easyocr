
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# =========================
# 1) Model loaders
# =========================
@st.cache_resource
def load_yolo(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)

@st.cache_resource
def load_reader():
    import easyocr
    return easyocr.Reader(['ko','en'], gpu=False)

# =========================
# 2) App UI
# =========================
st.set_page_config(page_title="간단 번호판 OCR", layout="wide")
st.title("번호판 인식 → 크롭(여백) → EasyOCR")

weights_path = st.text_input("YOLO 가중치 경로(.pt)", value="carplate_v11_yolo11n_70n.pt")
conf = st.slider("감지 신뢰도(conf)", 0.1, 0.9, 0.5, 0.05)
margin_px = st.slider("크롭 여백(px)", 0, 120, 2, 2)

uploaded = st.file_uploader("이미지 업로드 (jpg/png/bmp)", type=["jpg","jpeg","png","bmp"])

col1, col2 = st.columns(2)

if uploaded is not None and os.path.exists(weights_path):
    # 입력 이미지
    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)                       # RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # YOLO는 BGR/RGB 모두 가능하나 일관성 위해 BGR로

    # 모델
    yolo = load_yolo(weights_path)
    reader = load_reader()

    # 3) Detection
    results = yolo.predict(img_bgr, conf=conf, verbose=False)
    annotated = img_rgb.copy()

    crops = []           # [(crop_rgb, (x1,y1,x2,y2))]
    ocr_rows = []        # [{"plate_idx":i, "text":t, "conf":p}...]

    if len(results):
        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else []
        H, W = img_rgb.shape[:2]

        # 박스/크롭
        for i, (x1,y1,x2,y2) in enumerate(boxes_xyxy, start=1):
            # 여백 적용
            x1m = max(0, x1 - margin_px)
            y1m = max(0, y1 - margin_px)
            x2m = min(W, x2 + margin_px)
            y2m = min(H, y2 + margin_px)

            crop_rgb = img_rgb[y1m:y2m, x1m:x2m].copy()
            if crop_rgb.size == 0:
                continue

            # 시각화 (원본 위 박스)
            cv2.rectangle(annotated, (x1m,y1m), (x2m,y2m), (0,255,0), 2)
            cv2.putText(annotated, f"plate {i}", (x1m, max(0,y1m-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            crops.append((crop_rgb, (x1m,y1m,x2m,y2m)))

            # 4) EasyOCR (그레이스케일/보정 없이 바로)
            ocr_result = reader.readtext(crop_rgb)   # [(bbox, text, conf), ...]
            for (_, text, confv) in ocr_result:
                ocr_rows.append({
                    "plate_index": i,
                    "text": str(text),
                    "conf": float(confv)
                })

    # 5) 출력
    with col1:
        st.subheader("원본 + 감지 박스(여백 반영)")
        st.image(annotated, use_container_width=True)

        if crops:
            st.caption("각 크롭 결과")
            for idx, (crop_rgb, _) in enumerate(crops, start=1):
                st.image(crop_rgb, caption=f"Plate #{idx} crop", use_container_width=True)

    with col2:
        st.subheader("OCR 인식된 문자들 (그대로)")
        if ocr_rows:
            # 표로 보여주고, JSON도 보기 제공
            import pandas as pd
            df = pd.DataFrame(ocr_rows)
            st.dataframe(df, use_container_width=True)
            with st.expander("원시 JSON 보기"):
                st.json(ocr_rows)
        else:
            st.info("OCR 결과가 없습니다. conf를 낮추거나 여백/이미지를 조정하세요.")

else:
    if uploaded is None:
        st.info("이미지를 업로드하세요.")
    elif not os.path.exists(weights_path):
        st.warning("가중치 파일 경로가 올바르지 않습니다.")
