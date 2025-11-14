import cv2
import os

#khai bao camera va set kich thuoc khung hinh
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

#tep xml dung de phat hien khuon mat chinh dien
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#nhap id cho tung khuon mat
face_id = input('\nNhap thong tin sv: ')

print('\n Dang khoi tao camera....')

count = 0
while (True):

    ret, img= cam.read()
    #img= cv2.flip(img, -1)#lat anh theo chieu ngang va doc
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #phat hien khuon mat
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        #khoanh hinh chu nhat cho khuon mat
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count+=1

        cv2.imwrite('dataset/MSSV.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])

        cv2.imshow ('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count > 30:
        break

print('\n Thoat')

cam.release()
cv2.destroyAllWindows()

