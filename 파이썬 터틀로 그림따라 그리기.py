import cv2
import numpy as np
import turtle

# 이미지 로드 (그레이스케일)
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

# 1. 가우시안 블러로 노이즈 제거
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 2. Adaptive Thresholding 적용 (조명 변화에 강하게)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# 3. Morphological Closing (닫기 연산)으로 선 연결 보완
kernel = np.ones((3, 3), np.uint8)
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 혹은 Canny Edge 검출을 사용해볼 수도 있음 (파라미터는 이미지에 따라 조정)
# edges = cv2.Canny(blurred, 50, 150)

# 윤곽선 검출 (내부 윤곽선까지 모두)
contours, hierarchy = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# turtle 화면 설정 (이미지 중심 맞추기)
screen = turtle.Screen()
screen.setup(width=800, height=800)
t = turtle.Turtle()
t.speed(0)
t.penup()

# 이미지 크기와 오프셋 계산 (turtle 좌표 변환)
height, width = img.shape
offset_x = width // 2
offset_y = height // 2

# 각 윤곽선을 따라 turtle로 그리기
for contour in contours:
    if len(contour) == 0:
        continue
    x, y = contour[0][0]
    t.goto(x - offset_x, offset_y - y)
    t.pendown()
    for point in contour:
        x, y = point[0]
        t.goto(x - offset_x, offset_y - y)
    t.penup()

turtle.done()

