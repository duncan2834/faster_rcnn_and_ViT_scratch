# Faster R-CNN from Scratch
## Giới thiệu
Dự án này triển khai Faster R-CNN, một mô hình phát hiện đối tượng tiên tiến, được xây dựng từ đầu bằng Python và PyTorch. Mục tiêu là cung cấp một codebase dễ hiểu, dễ tùy chỉnh cho học tập và nghiên cứu về thị giác máy tính.
FasterRCNN-Pytorch
    -> VOC2007
        -> JPEGImages
        -> Annotations
    -> VOC2007-test
        -> JPEGImages
        -> Annotations
    -> tools
        -> train.py
        -> infer.py
        -> train_torchvision_frcnn.py
        -> infer_torchvision_frcnn.py
    -> config
        -> voc.yaml
    -> model
        -> faster_rcnn.py
    -> dataset
        -> voc.py
