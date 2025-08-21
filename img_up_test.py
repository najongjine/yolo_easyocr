import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# 해상도 자동 조절: 너무 작으면 키우고, 너무 크면 줄이기(색 유지)
def resize_keep_range(bgr, min_side=480, max_side=1600):
    h, w = bgr.shape[:2]
    scale = 1.0
    if min(h, w) < min_side:
        scale = min_side / float(min(h, w))
    elif max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    if scale != 1.0:
        bgr = cv2.resize(
            bgr, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )
    return bgr

# 간단 화이트밸런스(그레이월드) → 색감 안정
def grayworld_wb(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b *= (mean_gray / mean_b); g *= (mean_gray / mean_g); r *= (mean_gray / mean_r)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

# 사람 눈 기준 “자연스러운 컬러 화질 업” 파이프라인
def enhance_color(bgr,
                  min_side=480, max_side=1600, pad=4,
                  clahe_clip=2.0, sharp_amount=0.6,
                  denoise_h=5, sat_gain=1.07):
    # 1) 크기 자동 조절 + 약간 패딩(경계 보존)
    bgr = resize_keep_range(bgr, min_side=min_side, max_side=max_side)
    if pad > 0:
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # 2) 화이트밸런스
    bgr = grayworld_wb(bgr)

    # 3) L채널(밝기)만 선명/대비 보정 → 색 번짐 방지
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # (a) CLAHE로 로컬 대비 ↑ (문자/윤곽 살리되 과하게 X)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L = clahe.apply(L)

    # (b) 언샵 마스크로 선명도 ↑ (L만 적용)
    blur = cv2.GaussianBlur(L, (0, 0), 1.2)
    L = cv2.addWeighted(L, 1.0 + sharp_amount, blur, -sharp_amount, 0)

    lab = cv2.merge([L, A, B])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4) 가벼운 컬러 노이즈 제거(문자 보존)
    bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=denoise_h*5, sigmaSpace=denoise_h*4)

    # 5) 채도 아주 살짝 ↑ (자연스러운 느낌)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= sat_gain
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr

def main():
    parser = argparse.ArgumentParser(description="컬러 유지 화질 업그레이드(업스케일+선명+대비+노이즈)")
    parser.add_argument("--img", default="carplate4_1.jpg", help="입력 이미지 경로")
    parser.add_argument("--out", default=None, help="출력 경로(기본: *_enhanced_color.png)")
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--show", action="store_true", help="전/후 비교 표시")
    args = parser.parse_args()

    in_path = args.img
    out_path = args.out or (os.path.splitext(in_path)[0] + "_enhanced_color.png")

    bgr = cv2.imread(in_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {in_path}")

    out = enhance_color(bgr, min_side=args.min_side, max_side=args.max_side)
    cv2.imwrite(out_path, out)
    print("저장 완료 →", out_path)

    if args.show:
        rgb_in  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.title("Original"); plt.axis("off"); plt.imshow(rgb_in)
        plt.subplot(1,2,2); plt.title("Enhanced (Color)"); plt.axis("off"); plt.imshow(rgb_out)
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
