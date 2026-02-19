from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Xác định đường dẫn tuyệt đối cho file model
    # Lệnh os.getcwd() sẽ lấy thư mục hiện tại bạn đang đứng
    model_path = os.path.join(os.getcwd(), "yolov11n.pt")
    
    print(f"--- Đang tìm kiếm model tại: {model_path} ---")
    
    if not os.path.exists(model_path):
        print("LỖI: Vẫn không thấy file yolov11n.pt! Đang thử tải lại ép buộc...")
        # Lệnh này sẽ ép Ultralytics tải lại từ internet nếu không thấy file
        model = YOLO("yolo11n.pt") 
    else:
        print("Đã tìm thấy file! Bắt đầu huấn luyện...")
        model = YOLO(model_path)

    # 2. Huấn luyện
    model.train(
        data='data.yaml', 
        epochs=10, 
        imgsz=640, 
        device=0,      # Dùng RTX 5060
        workers=0,     # Quan trọng: Set về 0 để tránh lỗi đa luồng trên Windows
        batch=4        # Giảm batch size để đảm bảo không tràn VRAM trong lúc test
    )