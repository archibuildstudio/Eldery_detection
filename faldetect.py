import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import time

desired_fps = 60  # Target FPS
delay = 6.0 / desired_fps  # Target delay per frame (seconds)

model = YOLO("yolo11n.pt")
cap=cv2.VideoCapture('fall.mp4')
my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0

while True:
    start_time = time.time()  # Start timer

    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

#     ret,frame = cap.read()
#     count += 1
#     if count % 3 != 0:
#         continue
#     if not ret:
#        break

    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        h=y2-y1
        w=x2-x1
        thresh=h-w
        print(thresh) 
        if 'person' in c:
            if thresh <0:
                cvzone.putTextRect(frame,f'{"person_fall"}',(x1,y1),1,1)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            else:
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)    
                
    cv2.imshow("RGB", frame)

    processing_time = time.time() - start_time
    remaining_delay = max(1, int((delay - processing_time) * 1000))  # Convert to ms

    if cv2.waitKey(remaining_delay) & 0xFF == ord('q'):
        break

    # Break the loop if 'q' is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

cap.release()
cv2.destroyAllWindows()