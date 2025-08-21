import easyocr
import cv2
import matplotlib.pyplot as plt

# 1. OCR 리더 생성 (한국어 + 영어 지원 예시)
reader = easyocr.Reader(['ko', 'en'])  

# 2. 이미지 불러오기
image_path = 'carplate4_1_adaptive.png'   # 👉 여기에 이미지 경로 넣으세요
results = reader.readtext(image_path)

# 3. 결과 출력
for (bbox, text, prob) in results:
    print(f"인식된 글자: {text},  신뢰도: {prob:.2f}")

# 4. 이미지 위에 박스 그리기 (시각화)
img = cv2.imread(image_path)
for (bbox, text, prob) in results:
    # bbox = 네 꼭짓점 좌표
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    
    cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
    cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

# OpenCV는 BGR → Matplotlib은 RGB 이므로 변환
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
