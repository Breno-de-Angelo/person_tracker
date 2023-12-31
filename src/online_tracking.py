from ultralytics import YOLO


def main():
    model = YOLO("../models/yolov8n.pt")
    model.track(source=2, show=True)


if __name__ == '__main__':
    main()
