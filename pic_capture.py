import os
import sys
import cv2


def enter_new_user(data_path):
    new_user_name = input("Please enter your name：")
    print("Please look at the camera!")

    window_name = "Image Acquisition Area"
    camera_id = 0
    images_num = 200
    path = data_path + "\\" + new_user_name

    face_pic_capture(window_name, camera_id, images_num, path)

    print("Whether to add new personnel information(Press 'y' to continue and 'n' to exit)?")
    if input() == 'y':
        enter_new_user(data_path)
    else:
        pass


def face_pic_capture(window_name, camera_id, catch_pic_num, path_name):
    # 检查输入路径是否存在——不存在就创建
    CreateFolder(path_name)
    cv2.namedWindow(window_name)

    image_set = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    classifier = cv2.CascadeClassifier(
        r"D:\anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")

    num = 1
    while image_set.isOpened():
        flag, frame = image_set.read()
        if not flag:
            break

        image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRect = classifier.detectMultiScale(image_grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        if len(faceRect) > 0:  # 大于0则检测到人脸
            for facet in faceRect:
                x, y, w, h = facet

                if w > 100:
                    img_name = '{}/{}.jpg'.format(path_name, num)
                    image = frame[y - 10: y + h + 10, x - 5: x + w + 5]
                    cv2.imwrite(img_name, image)

                    cv2.rectangle(frame, (x - 5, y - 10), (x + w + 5, y + h + 10), (0, 0, 255), 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, "num:{}".format(num), (x + 30, y - 15), font, 1, (0, 250, 250), 4)

                    num += 1
                    if num > catch_pic_num:
                        break

        if num > catch_pic_num:
            break

        # 显示图像,按"Q"键中断采集过程
        cv2.imshow(window_name, frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭销毁所有窗口
    image_set.release()
    cv2.destroyAllWindows()


def CreateFolder(path):
    del_path_space = path.strip()
    del_path_tail = del_path_space.rstrip('\\')
    if not os.path.exists(del_path_tail):
        os.makedirs(del_path_tail)
        return True
    else:
        return False


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        enter_new_user(r'E:\Administrator\Pictures\data')
