import cv2
import numpy as np
import easyocr
from difflib import get_close_matches

REAL_ROOM_NAMES = [
    "ELECTRICAL", "CAFETERIA", "MEDBAY", "NAVIGATION", 
    "REACTOR", "SECURITY", "STORAGE", "COMMUNICATIONS", "DINING ROOM"
    "KITCHEN", "CORRIDORS", "WEAPONS"
]

# KHỞI TẠO READER TẠI ĐÂY (Để nó tồn tại trong phạm vi file utils.py)
# gpu=True để tận dụng sức mạnh của RTX 5060
reader = easyocr.Reader(['en'], gpu=True)

def get_dominant_color_v2(image_crop):
    if image_crop.size == 0: return "Unknown"

    # 1. Cắt lấy 2/3 phía trên để tránh nhiễu skin chân và sàn nhà
    h, w, _ = image_crop.shape
    upper_limit = int(h * 0.66)
    core_crop = image_crop[0:upper_limit, :]

    # 2. Chuyển sang HSV
    hsv = cv2.cvtColor(core_crop, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa dải màu (Có thể tinh chỉnh thêm tùy vào ánh sáng game)
    color_ranges = {
        "Red": [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
        "Blue": [([100, 150, 0], [140, 255, 255])],
        "Green": [([35, 100, 100], [85, 255, 255])],
        "Yellow": [([20, 100, 100], [30, 255, 255])],
        "Orange": [([10, 100, 100], [20, 255, 255])],
        "Black": [([0, 0, 0], [180, 255, 50])],
        "White": [([0, 0, 200], [180, 30, 255])]
    }

    max_pixels = 0
    dominant_color = "Unknown"

    # 3. Thuật toán bầu chọn đa số (Histogram)
    for color_name, ranges in color_ranges.items():
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_mask = cv2.bitwise_or(total_mask, mask)
        
        pixel_count = cv2.countNonZero(total_mask)
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            dominant_color = color_name

    return dominant_color

def fix_room_name(text):
    # Tìm từ gần giống nhất trong danh sách
    matches = get_close_matches(text, REAL_ROOM_NAMES, n=1, cutoff=0.4)
    if matches:
        return matches[0]
    return text # Nếu không giống cái nào thì giữ nguyên

def get_room_name(frame, coords):
    y1, y2, x1, x2 = coords
    room_crop = frame[y1:y2, x1:x2]
    if room_crop.size == 0: return "Unknown"

    # 1. Phóng to ảnh gấp 2 lần để chữ to hơn, OCR dễ đọc hơn
    room_crop = cv2.resize(room_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Chuyển sang ảnh xám và tăng độ tương phản cực đại
    gray = cv2.cvtColor(room_crop, cv2.COLOR_BGR2GRAY)
    
    # Sử dụng Threshold để tách hẳn chữ trắng ra khỏi nền (Dùng THRESH_OTSU để tự tính độ sáng)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = reader.readtext(thresh)
    
    if results:
        detected_text = results[0][1].upper()
        # Chuyển sang bước 2: Nắn chữ
        return fix_room_name(detected_text) 
    return "..."

def get_meeting_state(frame, coords):
    y1, y2, x1, x2 = coords
    state_crop = frame[y1:y2, x1:x2]
    if state_crop.size == 0: return "IDLE"

    # Tiền xử lý để OCR đọc chuẩn hơn
    gray = cv2.cvtColor(state_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    results = reader.readtext(thresh)
    if not results: return "IDLE"
    
    text = results[0][1].upper()

    # Phân loại dựa trên từ khóa (Keywords)
    if any(k in text for k in ["OPEN", "START", "REPORT"]):
        return "MEETING_OPENING"
    elif any(k in text for k in ["DISCUSS", "TALK", "CHAT"]):
        return "MEETING_DISCUSSION"
    elif any(k in text for k in ["VOTE", "SELECT", "TIME"]):
        return "MEETING_VOTING"
    
    return "MEETING_UNKNOWN"    