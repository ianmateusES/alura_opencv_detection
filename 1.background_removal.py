import numpy as np
import cv2

names_videos = ['Rua', 'Arco', 'Estradas', 'Peixes']
VIDEO = f"./data/{names_videos[0]}.mp4"

cap = cv2.VideoCapture(VIDEO)

frames_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in frames_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame, frame = cap.read()
    frames.append(frame)

median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
# cv2.imshow('Median Frame', median_frame)
# cv2.waitKey(0)
cv2.imwrite('./data/median_frame.jpg', median_frame)
