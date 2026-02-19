import os
import shutil
import urllib.parse

# --- ĐƯỜNG DẪN ---
IMG_SOURCE = r'C:\Users\PC\Desktop\Code\gooseBot\all_images'
LBL_SOURCE = r'C:\Users\PC\Desktop\Code\gooseBot\all_labels'
DEST_ROOT = r'C:\Users\PC\Desktop\Code\gooseBot\dataset_final'

# Tạo folder chuẩn YOLO
for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(DEST_ROOT, sub), exist_ok=True)

# Lấy danh sách ảnh hiện có (để so sánh)
img_dict = {os.path.splitext(f)[0]: f for f in os.listdir(IMG_SOURCE) if f.lower().endswith(('.jpg', '.png'))}

matched_count = 0
all_labels = [f for f in os.listdir(LBL_SOURCE) if f.endswith('.txt')]

for lbl_file in all_labels:
    # 1. Giải mã URL (biến %20 thành dấu cách, %5C thành \)
    decoded_name = urllib.parse.unquote(lbl_file)
    
    # 2. Lấy tên file gốc ở cuối đường dẫn
    # Ví dụ: ...\unlabeled_images\Goose Goose Duck_0111.txt -> Goose Goose Duck_0111
    original_name_with_ext = decoded_name.split('\\')[-1]
    original_name = os.path.splitext(original_name_with_ext)[0]

    # 3. Khớp và Copy
    if original_name in img_dict:
        # Xác định bộ (train hoặc val - tạm để 90/10)
        subset = 'train' if matched_count % 10 != 0 else 'val'
        
        # Copy Ảnh
        shutil.copy(os.path.join(IMG_SOURCE, img_dict[original_name]), 
                    os.path.join(DEST_ROOT, 'images', subset, img_dict[original_name]))
        # Copy Nhãn (đổi tên lại cho giống ảnh)
        shutil.copy(os.path.join(LBL_SOURCE, lbl_file), 
                    os.path.join(DEST_ROOT, 'labels', subset, original_name + '.txt'))
        
        matched_count += 1

print(f"--- THÀNH CÔNG! ---")
print(f"Đã giải mã và khớp được {matched_count} cặp ảnh-nhãn.")