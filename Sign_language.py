import math

import cv2
import cvzone
from ultralytics import YOLO

# sign_lang_class=['person','namaste','please_wait','ok','home','sad','love_you','thank_you']
sign_lang_class = [
    'hai',
    'love_you',
    'namaste',
    'no',
    'ok',
    'sad',
    'stop',
    'thanks'
]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO('../Yolo_weights/sign-lang-39-latest.pt')
model.to(device=0)
while True:
    success, video_org = cap.read()
    flipped_video = cv2.flip(video_org, 1)
    res = model(flipped_video, stream=True)
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > .4:
                cvzone.cornerRect(flipped_video, (x1, y1, w, h), l=30, colorR=(60, 7, 250), t=3, colorC=(46, 44, 48),
                                  rt=2)
                cvzone.putTextRect(flipped_video, f'{sign_lang_class[cls]} {conf}', (x1 + 20, y1 + 35), scale=.8,
                                   thickness=1, colorR=(0, 0, 0))
    cv2.imshow("Video", flipped_video)
    cv2.waitKey(2)
