from ultralytics import YOLO
import cv2 as cv
import cvzone
class Yolo:
    def __init__(self):
        #img = cv.imread(r"C:\Users\CPGT\Desktop\YoloV8\datasets\mc_1000\images\train2024\20240205_154659.jpg")
        #cv.imshow("Display window", img)
        #k = cv.waitKey(0)
        pass
    def train_model(self):
        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')
        results = model.train(data='config.yaml', epochs=25)
    def detect_item_in_model(self,imagem:str):
        img = cv.imread(imagem)
        model = YOLO(r'C:\Users\CPGT\Desktop\YoloV8\runs\segment\train5\weights\best.pt')
        results = model(imagem,stream=True)
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                cls = int(box.cls[0])
                conf = int(box.conf[0]*100)
                if conf >= 30:
                    cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),4)
                    cvzone.putTextRect(img,'MC-1000 {}%'.format(conf),(x1,y1-5))
        img = cv.resize(img,(640,640))

        cv.imshow("PT-76", img)
        k = cv.waitKey(0)


if __name__ == '__main__':
    yolo = Yolo()
    yolo.detect_item_in_model(imagem=r'C:\Users\CPGT\Desktop\YoloV8\testandoTreino.jpg')