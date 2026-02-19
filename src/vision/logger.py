import json
import os
from datetime import datetime

def log_event(color, x, y, room_name="Unknown"):
    log_path = "data/logs/raw_vision.json"
    
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "color": color,
        "position": [x, y],
        "room": room_name
    }
    
    # Đọc dữ liệu cũ và ghi thêm dữ liệu mới
    data = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try: data = json.load(f)
            except: data = []
            
    data.append(event)
    # Giới hạn dung lượng log để tránh nặng máy (chỉ giữ 500 sự kiện gần nhất)
    if len(data) > 500: data.pop(0)
    
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=4)