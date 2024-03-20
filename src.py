from ultralytics import YOLO
import cv2
from deepface import DeepFace
video=cv2.VideoCapture(0)
faceModel=YOLO('yolov8n-face.pt')
while 1:
        _,frame=video.read()

        face_result=faceModel.predict(frame,conf=0.4)

        for info in face_result:
                parameters=info.boxes
                for box in parameters:
                        x1,y1,x2,y2=box.xyxy[0]
                        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                        h,w=y2-y1,x2-x1
                        flag=frame[y1:y1+h,x1:x1+w]
                        objs=DeepFace.analyze(flag,actions=['emotion'],enforce_detection=False)
                        maxRatio=0
                        emo=""
                        for emos in objs[0]['emotion']:
                                if(maxRatio<objs[0]['emotion'][emos]):
                                        maxRatio=objs[0]['emotion'][emos]
                                        emo=emos
                        frame=cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),3)
                        frame=cv2.putText(frame,emo,(x1,y1+h+25),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),1,lineType=cv2.LINE_AA,bottomLeftOrigin=None)
        cv2.imshow("frame",frame)

        if cv2.waitKey(1) & 0xff==ord('q'):
                break
video.release()

cv2.destroyAllWindows()
