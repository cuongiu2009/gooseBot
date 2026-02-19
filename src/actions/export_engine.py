from ultralytics import YOLO

def export_to_engine():
    # 1. Nạp file .pt xịn nhất từ folder số 9 của bạn
    # Hãy đảm bảo đường dẫn này đúng với thực tế trên máy bạn
    model = YOLO(r'C:\Users\PC\Desktop\Code\gooseBot\runs\detect\goose_final_model9\weights\best.pt')

    print("--- Đang bắt đầu quá trình biên dịch TensorRT ---")
    print("Lưu ý: Quá trình này có thể mất 3-5 phút. Đừng tắt máy!")

    # 2. Thực hiện Export
    # format='engine': Xuất ra định dạng TensorRT
    # device=0: Ép sử dụng GPU RTX 5060
    # half=True: Sử dụng FP16 để tăng tốc gấp đôi và giảm nhiệt độ
    # imgsz=640: Giữ nguyên kích thước ảnh đã huấn luyện
    model.export(format='engine', device=0, half=True, imgsz=640)

    print("--- Hoàn thành! ---")
    print("Bạn sẽ thấy file 'best.engine' xuất hiện cùng thư mục với file 'best.pt'")

if __name__ == '__main__':
    export_to_engine()