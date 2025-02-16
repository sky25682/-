import cv2  # OpenCV를 사용하기 위한 모듈
import numpy as np  # NumPy를 사용하기 위한 모듈
import pytesseract  # Tesseract OCR을 사용하기 위한 모듈
import matplotlib.pyplot as plt  # 결과를 시각화하기 위한 모듈
import pandas as pd
from PIL import ImageGrab
from sklearn.preprocessing import MinMaxScaler
import pyautogui
import requests
import json
import base64


left_1 = 255
top = 218
right_1 = 815
bottom = 540

# 지정된 영역을 캡처
screenshot = ImageGrab.grab(bbox=(left_1, top, right_1, bottom))
image_path = "apple_live.png"  # 여기에 이미지 파일 경로 입력
# 캡처한 이미지를 저장 (선택 사항)
screenshot.save("apple_live.png")



API_KEY = 'API_KEY'
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"


# 이미지 파일을 Base64로 인코딩하는 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 이미지 경로 설정

encoded_image = encode_image(image_path)

# 요청 데이터 설정 (텍스트 + 이미지)
data = {
    "contents": [{
        "parts": [
            {"text": "사진에 있는 숫자들을 파이썬 이차원 배열로 만들어줘. 배열만 주고 다른 말은 하지마. 그리고 배열의 이름은 apple_cor이거로해."},  # 텍스트 질문
            {"inline_data": {
                "mime_type": "image/jpeg",  # 이미지 타입 (jpeg, png 등)
                "data": encoded_image  # Base64 인코딩된 이미지 데이터
            }}
        ]
    }]
}

# 요청 헤더 설정
headers = {
    "Content-Type": "application/json"
}

# POST 요청 보내기
response = requests.post(URL, headers=headers, data=json.dumps(data))
json_data = response.json()
strs = json_data['candidates'][0]['content']['parts'][0]['text']
print(strs)

apple_cor = eval(strs[22:-4])




print("before arr")
for i in apple_cor:
 print(i)
# 행(row) 수
rows = len(apple_cor)
# 열(column) 수 (첫 번째 행의 길이 기준)
cols = len(apple_cor[0]) if rows > 0 else 0

def down(x,y,is_ten):
  is_ten = 0
  cnt_i = 0
  coordinate = []
  while(is_ten<10 and x + cnt_i < 10):
    is_ten += apple_cor[x+ cnt_i][y]
    if (apple_cor[x+ cnt_i][y] != 0):
      coordinate.append((x+ cnt_i,y ))
    cnt_i += 1
  return coordinate, is_ten

def up(x,y,is_ten):
  is_ten = 0
  cnt_i = 0
  coordinate = []
  while(is_ten<10 and x - cnt_i >= 0 ):
    is_ten += apple_cor[x - cnt_i][y]
    if (apple_cor[x - cnt_i][y] != 0):
      coordinate.append((x - cnt_i,y ))
    cnt_i += 1
  return coordinate, is_ten

def left(x,y,is_ten):
  
  is_ten = 0
  cnt_i = 0
  coordinate = []
  while(is_ten<10  and y - cnt_i >= 0):
    is_ten += apple_cor[x ][y - cnt_i]
    if (apple_cor[x ][y - cnt_i] != 0):
      coordinate.append((x,y - cnt_i))
    cnt_i += 1
  return coordinate, is_ten

def right(x,y,is_ten):
  print(x,y)
  is_ten = 0
  cnt_i = 0
  coordinate = []
  while(is_ten<10 and y + cnt_i < 17):
    is_ten += apple_cor[x][y + cnt_i]
    if (apple_cor[x][y + cnt_i] != 0):
      coordinate.append((x,y + cnt_i))
    cnt_i += 1
  return coordinate, is_ten


def check(coordinate, loc1, loc2):
  base_x = 255
  base_y = 221
  for x,y in coordinate:
    apple_cor[x][y] = 0
  # 시작 위치로 마우스 이동 (x, y 좌표)
  y,x  = loc1
  
  start_x, start_y = base_x + x * 33.5  , base_y + y * 33.5
  #위치를 고정으로 시켜서 하자
  pyautogui.moveTo(start_x, start_y,duration=0.3)

  # 마우스를 특정 위치로 드래그 (드래그할 끝 좌표 x, y)
  loc2_y,loc2_x = loc2
  end_x, end_y = base_x + loc2_x * 33.5 + 23  , base_y + loc2_y * 33.5 + 23
  # 드래그 시작
  
  
  pyautogui.dragTo(end_x, end_y, duration=0.6, tween=pyautogui.easeInOutQuad,button='left')  # 드래그할 위치로 이동 (duration은 이동 속도)
  

#x,y 는 행, 열
def find_10(x,y):
  coordinate = []
  is_ten = 0
  if(x != 9):
    coordinate, is_ten = up(x,y,is_ten)
    if is_ten == 10:
      check(coordinate, coordinate[-1],coordinate[0] )
      return 1
  if(x != 0):
    coordinate, is_ten = down(x,y,is_ten)
    if is_ten == 10:
      check(coordinate, coordinate[0], coordinate[-1])
      return 1

  if(y != 16):
    coordinate, is_ten = right(x,y,is_ten)
    if is_ten == 10:
      check(coordinate, coordinate[0], coordinate[-1])
      return 1

  if(y != 0):
    coordinate, is_ten = left(x,y,is_ten)
    if is_ten == 10:
      check(coordinate, coordinate[-1] ,coordinate[0])
      return 1

for x in range(rows): # 10
  print(f"rows :: {x}")
  for y in range(cols): # 17
    find_10(x,y)

print("after arr")
for i in apple_cor:
  print(i)

