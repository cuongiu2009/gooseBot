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

def get_dominant_color(image_crop):
    if image_crop.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    center_region = hsv[int(h*0.4):int(h*0.7), int(w*0.3):int(w*0.7)]
    avg_h = np.mean(center_region[:, :, 0])
    avg_s = np.mean(center_region[:, :, 1])
    avg_v = np.mean(center_region[:, :, 2])

    if avg_v < 50: return "Black"
    if avg_s < 40 and avg_v > 150: return "White"
    if (avg_h < 7) or (avg_h > 170): return "Red"
    elif 7 <= avg_h < 18: return "Orange"
    elif 18 <= avg_h < 35: return "Yellow"
    elif 35 <= avg_h < 85: return "Green"
    elif 85 <= avg_h < 100: return "Cyan"
    elif 100 <= avg_h < 130: return "Blue"
    elif 130 <= avg_h < 150: return "Purple"
    elif 150 <= avg_h < 170: return "Pink"
    return "Unknown"

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