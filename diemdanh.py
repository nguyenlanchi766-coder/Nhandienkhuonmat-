import cv2
import os
import csv
import datetime

# ================== LOAD MODEL SAU KHI TRAIN ==================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Haarcascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ================== CAMERA ==================
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

font = cv2.FONT_HERSHEY_SIMPLEX

# ================== FILE CSV ĐIỂM DANH ==================
attendance_file = "diemdanh.csv"

# Tạo file CSV nếu chưa tồn tại
if not os.path.exists(attendance_file):
    with open(attendance_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["MSSV", "Thời điểm"])

# Bộ nhớ lưu MSSV đã điểm danh để tránh trùng
checked_ids = set()

print("\n===== HỆ THỐNG ĐIỂM DANH BẰNG KHUÔN MẶT =====\n")

# ================== VÒNG LẶP NHẬN DIỆN ==================
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Vẽ khung khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Nhận diện ID
        id_predict, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Confidence càng thấp càng tốt
        if confidence < 70:
            mssv = str(id_predict)
            cv2.putText(frame, "MSSV: " + mssv, (x, y-10), font, 1, (0, 255, 0), 2)

            # Ghi điểm danh nếu MSSV chưa được điểm danh
            if mssv not in checked_ids:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(attendance_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([mssv, now])

                print(f"--> Đã điểm danh MSSV {mssv} lúc {now}")
                checked_ids.add(mssv)

        else:
            cv2.putText(frame, "Unknown", (x, y-10), font, 1, (0, 0, 255), 2)

    cv2.imshow("He thong diem danh", frame)

    # Nhấn ESC để thoát
    k = cv2.waitKey(10)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()

print("\n===== KẾT THÚC - FILE CSV ĐÃ ĐƯỢC LƯU =====")
print("File: diemdanh.csv")
