import cv2  # OpenCV를 사용하기 위한 모듈
import numpy as np  # NumPy를 사용하기 위한 모듈
import pytesseract  # Tesseract OCR을 사용하기 위한 모듈
import matplotlib.pyplot as plt  # 결과를 시각화하기 위한 모듈
import pandas as pd
from PIL import ImageGrab
from sklearn.preprocessing import MinMaxScaler
import pyautogui
def invert_colors(img):
    # 회색 영역(128~200)은 흰색으로 바꾸고, 나머지는 흰색을 검정으로 바꾼다.
    inverted_img = np.where(img < 128, 255, 0)  # 흰색 영역을 검정으로 바꿈
    inverted_img = np.where((img >= 128) & (img <= 200), 255, inverted_img)  # 회색 영역을 흰색으로 바꿈
    return inverted_img.astype(np.uint8)
# 캡처할 영역의 좌표 (왼쪽, 위, 오른쪽, 아래)
# 예: 왼쪽(100, 100)에서 오른쪽(500, 400)까지 캡처


left_1 = 252
top = 211
right_1 = 815
bottom = 540

# 지정된 영역을 캡처
#screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))

# 캡처한 이미지를 저장 (선택 사항)
#screenshot.save("apple_live.png")

print("Screenshot capture")

numbers = []
img_path = "apple_live.png"
image = cv2.imread(img_path)

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Canny 엣지 검출
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
# 윤곽 추출
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
x_leng, y_leng = 15, 15
n = 3
coordinates = []
apple_x_y = []
print("Edge extraction")

# 윤곽 그리기
for contour in contours:
  x, y , w, h = cv2.boundingRect(contour)
  
  if 4 < w < x_leng and 4 < h < y_leng:
    #이미지 자르기
    

    roi = image[y-n:y+n+h,x-n:x+w+n]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)


    height, width = roi_gray.shape[:2]
    new_width = width * 3
    new_height = height * 3
    # 크기 조정
    resized_image = cv2.resize(roi_thresh, (new_width, new_height))

    
    eroded_image = invert_colors(resized_image)
    #cv2_imshow(eroded_image) # 테스트용

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(eroded_image, config=custom_config)
    try:
      if int(text) >=0 and int(text) < 10 :
        numbers.append(int(text))
        coordinates.append((int(text),x,y,x,y,w,h))
        #print(x,y)
        #print("Extracted Text:", text) # 테스트용
    except:
      coordinates.append((8,x,y,x,y,w,h))
      numbers.append(8)
      #print("Extracted Text: 8") # 테스트용

print("cut number")

for (_, x,y,_,_,_,_) in coordinates:
  plt.scatter(x,y,color='red',s=10)
#plt.show()
# 결과 출력

df = pd.DataFrame(coordinates)
df.columns = ["weight", "x", "y","img_x","img_y","img_w","img_h"]

scalerX = MinMaxScaler((0,16))
scalerY = MinMaxScaler((0,9))

df['x'] = scalerX.fit_transform(df["x"].values.reshape(-1,1))
df['y'] = scalerY.fit_transform(df["y"].values.reshape(-1,1))

df = df.round(0).astype(int)

plt.xlim([-1, 17])      # X축의 범위: [xmin, xmax]
plt.ylim([-1, 10])     # Y축의 범위: [ymin, ymax]
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
plt.yticks([0,1,2,3,4,5,6,7,8,9])

plt.scatter(df['x'], df['y'])
#plt.show()
#df.head()
print("nomalization plt")

apple = np.zeros((10,17), dtype=tuple)
for i in range(len(df)):
    n,x,y,img_x,img_y,img_w,img_h = df.loc[i]
    apple[y][x] = (i, n,img_x,img_y,img_w,img_h)
apple_cor = []
for i in apple:
    temp= []
    temp_img_cor = []
    for j in i:
        temp.append(j[1])
        temp_img_cor.append((j[2],j[3],j[4],j[5]))
     #   print(j[1], end=" ")
    apple_cor.append(temp)
    apple_x_y.append(temp_img_cor)
    #print("")

temp_apple = apple_cor
