# Faster R-CNN from Scratch
## Giới thiệu
Dự án này triển khai Faster R-CNN, một mô hình phát hiện đối tượng tiên tiến, được xây dựng từ đầu bằng Python và PyTorch. Mục tiêu là cung cấp một codebase dễ hiểu, dễ tùy chỉnh cho học tập và nghiên cứu về thị giác máy tính.
## Tính năng
- Triển khai hoàn toàn từ đầu, không sử dụng thư viện như torchvision cho Faster R-CNN.
- Hỗ trợ huấn luyện và đánh giá trên tập dữ liệu tùy chỉnh.
- Bao gồm các thành phần chính: Region Proposal Network (RPN), ROI Pooling, Classification Head.
- Tối ưu hóa hiệu suất với PyTorch.
## Model
faster_rcnn_scratch/
├── data/               # Chứa dữ liệu và tập dữ liệu (ví dụ: COCO)
├── models/             # Định nghĩa mô hình Faster R-CNN
├── utils/              # Các hàm tiện ích (dataloader, visualization,...)
├── train.py            # Script để huấn luyện mô hình
├── evaluate.py         # Script để đánh giá mô hình
├── predict.py          # Script để dự đoán trên ảnh mới
├── requirements.txt    # Danh sách các thư viện cần thiết
└── README.md           # Tài liệu hướng dẫn dự án
