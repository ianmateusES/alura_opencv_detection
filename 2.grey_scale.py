import numpy as np
import cv2
from time import sleep

names_videos = ['Rua', 'Arco', 'Estradas', 'Peixes']
VIDEO = f"./data/{names_videos[0]}.mp4"
delay = 10

cap = cv2.VideoCapture(VIDEO)

frames_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in frames_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame, frame = cap.read()
    frames.append(frame)

median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
gray_median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Cinza', gray_median_frame)
#cv2.waitKey(0)
cv2.imwrite('./data/median_frame_cinza.jpg', gray_median_frame)

while (True):
    tempo = float(1 / delay)
    sleep(tempo)
    
    has_frame, frame = cap.read()

    if not has_frame:
        print('Acabou os frames')
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    d_frame = cv2.absdiff(frame_gray, gray_median_frame)
    th, dframe = cv2.threshold(d_frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('Frames em Cinza', frame_gray)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
