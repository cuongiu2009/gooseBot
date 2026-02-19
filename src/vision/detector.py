import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import os
import yaml
# Đảm bảo bạn đã thêm get_meeting_state vào utils.py
from utils import get_dominant_color, get_room_name, get_meeting_state

# --- THIẾT LẬP ĐƯỜNG DẪN HỆ THỐNG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'settings.yaml')

# Đọc cấu hình với encoding utf-8 để tránh lỗi Unicode
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Khong tim thay file settings.yaml tai: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

MODEL_PATH = os.path.join(BASE_DIR, config['model_path'])
TARGET_FPS = config.get('target_fps', 30)
TIME_PER_FRAME = 1.0 / TARGET_FPS

# Khởi tạo AI Model
model = YOLO(MODEL_PATH, task='detect')

def start_detector():
    # BƯỚC 1: CHỌN CÁC VÙNG QUAN SÁT (ROI)
    with mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        print("--- [1/3] QUÉT VÙNG CỬA SỔ GAME ---")
        roi_game = cv2.selectROI("1. Vung Game (Bam ENTER sau khi quet)", screenshot)
        cv2.destroyWindow("1. Vung Game (Bam ENTER sau khi quet)")
        
        print("--- [2/3] QUÉT VÙNG TÊN PHÒNG ---")
        roi_room = cv2.selectROI("2. Vung Ten Phong (Bam ENTER sau khi quet)", screenshot)
        cv2.destroyWindow("2. Vung Ten Phong (Bam ENTER sau khi quet)")

        print("--- [3/3] QUÉT VÙNG TRẠNG THÁI HỌP ---")
        roi_state = cv2.selectROI("3. Vung Trang Thai Hop (Bam ENTER sau khi quet)", screenshot)
        cv2.destroyWindow("3. Vung Trang Thai Hop (Bam ENTER sau khi quet)")

        x_g, y_g, w_g, h_g = roi_game
        x_r, y_r, w_r, h_r = roi_room
        x_s, y_s, w_s, h_s = roi_state

        if w_g == 0 or w_r == 0 or w_s == 0:
            print("❌ Loi: Ban chua quet du 3 vung can thiet!")
            return

        # Thiet lap monitor cho mss và tọa độ tương đối
        monitor_game = {"top": y_g, "left": x_g, "width": w_g, "height": h_g}
        room_coords = (y_r - y_g, y_r - y_g + h_r, x_r - x_g, x_r - x_g + w_r)
        state_coords = (y_s - y_g, y_s - y_g + h_s, x_s - x_g, x_s - x_g + w_s)

    print(f"--- BOT DANG QUAN SAT TAI {TARGET_FPS} FPS ---")
    frame_count = 0
    current_room = "Unknown"
    meeting_state = "IDLE"

    # BƯỚC 2: VÒNG LẶP XỬ LÝ CHÍNH
    with mss() as sct:
        while True:
            start_time = time.time()

            # Chụp ảnh và ép kiểu dữ liệu liên tục để tránh lỗi OpenCV PutText
            img_raw = np.array(sct.grab(monitor_game))
            frame = np.ascontiguousarray(img_raw[:, :, :3])

            # 1. Nhận diện vật thể (Ngỗng)
            results = model.predict(frame, conf=config['conf_threshold'], device=0, half=True, verbose=False)

            # 2. Cập nhật Môi trường & Trạng thái (mỗi 0.5 giây)
            if frame_count % 15 == 0:
                current_room = get_room_name(frame, room_coords)
                meeting_state = get_meeting_state(frame, state_coords)
                # In debug ra Terminal
                print(f"DEBUG >> Room: {current_room} | State: {meeting_state}")

            # 3. Vẽ kết quả nhận diện lên frame
            if len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Cắt ảnh con ngỗng để nhận diện màu
                    goose_crop = frame[y1:y2, x1:x2]
                    color = get_dominant_color(goose_crop)
                    
                    # Vẽ khung và tên màu
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. Vẽ HUD (Heads-up Display) thông tin tổng quát
            cv2.putText(frame, f"ROOM: {current_room}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"STATE: {meeting_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Hiển thị cửa sổ Debug
            cv2.imshow("GooseBot Vision System", frame)

            frame_count += 1

            # Khóa FPS để ổn định tài nguyên CPU/GPU
            elapsed = time.time() - start_time
            if elapsed < TIME_PER_FRAME:
                time.sleep(TIME_PER_FRAME - elapsed)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_detector()