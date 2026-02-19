from ultralytics import YOLO
import os

# 1. Load cái model 50 tấm bạn vừa train
model = YOLO('best.pt') 

# 2. Đường dẫn folder ảnh chưa gán nhãn và folder xuất kết quả
input_dir = 'unlabeled_images'
output_dir = 'datasets/v2/labels' # Lưu vào folder dataset mới

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Chạy dự đoán và lưu nhãn
# conf=0.4: Chỉ lấy những gì AI chắc chắn trên 40% để tránh rác
results = model.predict(source=input_dir, save_txt=True, conf=0.4, project='auto_label_results')

print(f"--- Đã quét xong! Kiểm tra các file .txt tại thư mục 'auto_label_results' ---")