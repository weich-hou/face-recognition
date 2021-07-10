import os
import sys
import cv2

from model_train import train
from model_train import Model
from pic_capture import enter_new_user
SWITCH = 0

data_path = r'E:\Administrator\Pictures\data'
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    # train(data_path)
    model.load_model(file_path=r'../face_recognition/model/train_model.h5')

    images = cv2.VideoCapture(0)

    while True:
        if SWITCH == 1:
            enter_new_user(data_path)
            train(data_path)
        else:
            flag, frame = images.read()
            if flag:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue

            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(
                r"D:\anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    image = frame[x - 5: x + w + 5, y - 10: y + h + 10]
                    faceID = model.face_predict(image, 64, 64)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 255), thickness=2)

                    for i in range(len(os.listdir(data_path))):
                        if i == faceID:
                            cv2.putText(frame, os.listdir(data_path)[i],
                                        (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            print("Welcome, Mr." + os.listdir(data_path)[i])
                            break
                        else:
                            print("Unrecognized, please try again!\n")
                            continue
                # break

            cv2.imshow("who am I", frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    images.release()
    cv2.destroyAllWindows()



