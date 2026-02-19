import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import os

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
MODEL_PATH = r'C:\Users\PC\Desktop\Code\gooseBot\runs\detect\goose_final_model9\weights\best.engine'
SAVE_DIR = r'C:\Users\PC\Desktop\Code\gooseBot\retrain_data'

TARGET_FPS = 30
TIME_PER_FRAME = 1.0 / TARGET_FPS

# Khoảng Confidence mà AI đang "phân vân" để chụp ảnh lại
LEARNING_RANGE = (0.35, 0.65) 
# Thời gian chờ giữa 2 lần chụp ảnh (giây) để tránh trùng lặp dữ liệu
SAVE_COOLDOWN = 1.0 

# Tạo thư mục lưu trữ nếu chưa có
os.makedirs(SAVE_DIR, exist_ok=True)

# Nạp model (Bắt buộc dùng task='detect' cho file .engine)
model = YOLO(MODEL_PATH, task='detect')

def select_roi():
    """Cho phép người dùng quét vùng màn hình để theo dõi"""
    with mss() as sct:
        # Chụp toàn bộ màn hình chính
        screenshot = np.array(sct.grab(sct.monitors[1]))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        window_name = "HƯỚNG DẪN: Quét vùng Game rồi nhấn ENTER"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        roi = cv2.selectROI(window_name, screenshot, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        return roi

def start_bot():
    # 1. Chọn vùng quét
    x, y, w, h = select_roi()
    if w == 0 or h == 0:
        print("Chưa chọn vùng quét. Đang thoát...")
        return

    monitor = {"top": y, "left": x, "width": w, "height": h}
    last_save_time = 0
    
    print(f"--- ĐANG GIÁM SÁT TẠI: {x, y, w, h} ---")
    print(f"Khóa mục tiêu 30 FPS. Nhấn 'q' để dừng.")

    with mss() as sct:
        while True:
            start_frame_time = time.time()

            # 2. Chụp vùng màn hình đã chọn
            img = np.array(sct.grab(monitor))
            frame = img[:, :, :3]  # Chuyển về BGR cho OpenCV và YOLO

            # 3. Dự đoán với TensorRT (Half Precision cho RTX 5060)
            results = model.predict(frame, conf=0.25, device=0, half=True, verbose=False)

            current_time = time.time()
            
            # Duyệt qua các kết quả tìm thấy
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                """
                # CƠ CHẾ TỰ HỌC: Lưu ảnh khi AI phân vân + Có Cooldown
                if LEARNING_RANGE[0] < conf < LEARNING_RANGE[1]:
                    if current_time - last_save_time > SAVE_COOLDOWN:
                        timestamp = int(current_time * 1000)
                        file_base = os.path.join(SAVE_DIR, f"study_{label}_{timestamp}")
                        
                        # Lưu ảnh sạch và file nhãn dự đoán (.txt)
                        cv2.imwrite(f"{file_base}.jpg", frame)
                        results[0].save_txt(f"{file_base}.txt")
                        
                        last_save_time = current_time
                        print(f"📸 Đã lưu mẫu học: {label} ({conf:.2f})")
                """
            # 4. Hiển thị Vision (Kéo cửa sổ này ra góc khác để tránh Mirror Effect)
            annotated_frame = results[0].plot()
            
            # Tính FPS thực tế để hiển thị
            actual_fps = 1.0 / (time.time() - start_frame_time + 1e-5)
            cv2.putText(annotated_frame, f"FPS: {int(actual_fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("GooseBot - Vision Monitoring", annotated_frame)

            # 5. ĐIỀU TIẾT KHUNG HÌNH (Khóa 30 FPS)
            elapsed = time.time() - start_frame_time
            if elapsed < TIME_PER_FRAME:
                time.sleep(TIME_PER_FRAME - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        start_bot()
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")