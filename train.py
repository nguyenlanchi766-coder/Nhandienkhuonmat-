import cv2 
import numpy as np
import os
from PIL import Image

# Thư mục chứa các mẫu khuôn mặt đã được gắn nhãn (labeled face samples)
Path = 'dataset'
# Thư mục để lưu file model đã train
TRAINER_DIR = 'trainer'
if not os.path.exists(TRAINER_DIR):
    os.makedirs(TRAINER_DIR)

# Sử dụng thuật toán LBPH (Local Binary Pattern Histograms) để nhận dạng
recognizer = cv2.face.LBPHFaceRecognizer_create() 
# Sử dụng Haar Cascade để phát hiện khuôn mặt
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    # Tạo một danh sách các đường dẫn đầy đủ đến tất cả các tệp trong thư mục Path
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        # Bỏ qua các tệp không phải là ảnh
        if 'desktop.ini' in image_path.lower():
            continue
        
        # Mở ảnh bằng PIL và chuyển sang ảnh grayscale (L)
        PIL_img = Image.open(image_path).convert("L")
        # Chuyển đổi ảnh PIL sang mảng NumPy 8-bit
        img_numpy = np.array(PIL_img, 'uint8')

        # Tách ID từ tên tệp (ví dụ: 'MSSV.2913236137_PhanMyHanh.20.jpg' -> id = 2913236137)
        try:
            # Lấy phần cuối cùng của đường dẫn, sau đó tách theo dấu '.' và lấy phần tử thứ 2
            face_id = int(os.path.split(image_path)[-1].split(".")[1][1:10])
        except IndexError:
             # In ra lỗi nếu tên tệp không đúng định dạng và bỏ qua
            print(f"Lỗi: Tên tệp '{os.path.basename(image_path)}' không đúng định dạng. Cần phải có ID (vd: user.1.1.jpg). Bỏ qua tệp này.")
            continue
            
        # Phát hiện khuôn mặt trong ảnh grayscale
        faces = detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5)

        # Trích xuất và lưu mẫu khuôn mặt (facesamples)
        for (x, y, w, h) in faces:
            # Cắt phần khuôn mặt và thêm vào danh sách mẫu
            face_samples.append(img_numpy[y:y+h, x:x+w])
            # Thêm ID tương ứng
            ids.append(face_id)

    return face_samples, ids


print ('\n--- BẮT ĐẦU TRAINING DỮ LIỆU ---')
# Lấy các mẫu khuôn mặt và nhãn ID
faces, ids = get_images_and_labels(Path)

if not faces:
    print("\nLỖI: Không tìm thấy khuôn mặt nào hợp lệ trong thư mục 'dataset'. Vui lòng kiểm tra lại ảnh.")
else:
    # Bắt đầu quá trình huấn luyện mô hình
    recognizer.train(faces, np.array(ids))

    # Lưu mô hình đã train vào tệp trainer.yml
    trainer_file_path = os.path.join(TRAINER_DIR, 'trainer.yml')
    recognizer.write(trainer_file_path)

    # In ra số lượng ID duy nhất đã được train
    unique_ids = np.unique(ids)
    print('\n [{0}] ID khuôn mặt duy nhất đã được train thành công.'.format(len(unique_ids)))
    print('Dữ liệu đã lưu vào: {0}. Thoát chương trình.'.format(trainer_file_path))