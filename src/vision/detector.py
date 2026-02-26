import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import os
import yaml
from utils import get_dominant_color_v2, get_room_name, get_meeting_state

# --- LOAD CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'settings.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model = YOLO(os.path.join(BASE_DIR, config['model_path']), task='detect')

def start_detector():
    # 1. Khởi tạo bộ nhớ ID bên trong hàm để tránh lỗi NameError
    track_history = {} 

    with mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # Chọn 3 vùng ROI (Quét Game -> Quét Phòng -> Quét Trạng thái)
        roi_g = cv2.selectROI("1. Vung Game", screenshot); cv2.destroyAllWindows()
        roi_r = cv2.selectROI("2. Vung Phong", screenshot); cv2.destroyAllWindows()
        roi_s = cv2.selectROI("3. Vung Trang Thai", screenshot); cv2.destroyAllWindows()

        mon_g = {"top": roi_g[1], "left": roi_g[0], "width": roi_g[2], "height": roi_g[3]}
        r_coords = (roi_r[1]-roi_g[1], roi_r[1]-roi_g[1]+roi_r[3], roi_r[0]-roi_g[0], roi_r[0]-roi_g[0]+roi_r[2])
        s_coords = (roi_s[1]-roi_g[1], roi_s[1]-roi_g[1]+roi_s[3], roi_s[0]-roi_g[0], roi_s[0]-roi_g[0]+roi_s[2])

    frame_count = 0
    room, state = "Unknown", "IDLE"

    with mss() as sct:
        while True:
            img = np.array(sct.grab(mon_g))
            frame = np.ascontiguousarray(img[:, :, :3])

            # CHẠY TRACKING
            # device=0: Sử dụng GPU đầu tiên (RTX 5060)
            # half=True: Sử dụng Half-Precision (FP16) cho TensorRT (nhanh hơn & tiết kiệm VRAM)
            results = model.track(frame, persist=True, conf=0.5, iou=0.5, device=0, half=True, verbose=False)
            names = model.names

            if frame_count % 15 == 0:
                room = get_room_name(frame, r_coords)
                state = get_meeting_state(frame, s_coords)

            # XỬ LÝ BOXES & ID
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, t_id, c_id in zip(boxes, ids, clss):
                    label = names[c_id].upper()
                    x1, y1, x2, y2 = box

                    if t_id not in track_history or track_history[t_id] == "Unknown":
                        color = get_dominant_color_v2(frame[y1:y2, x1:x2])
                        if color != "Unknown": track_history[t_id] = color

                    f_color = track_history.get(t_id, "Scanning...")
                    b_color = (0, 255, 0) if label == "GOOSE" else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 2)
                    cv2.putText(frame, f"{label} #{t_id}: {f_color}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 2)

            # HUD
            cv2.putText(frame, f"ROOM: {room} | STATE: {state}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("GooseBot Vision", frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_detector()