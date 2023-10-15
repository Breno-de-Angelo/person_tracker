from ultralytics import YOLO
import cv2
import threading


def save_tracker(in_file, out_file, model, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI or 'mp4v' for MP4
    out = cv2.VideoWriter(out_file, fourcc, fps, frame_size)
    for result in model.track(source=in_file, stream=True):
        out.write(result.plot())
    out.release()


def main():
    model1 = YOLO("../models/yolov8n.pt")
    model2 = YOLO("../models/yolov8n.pt")
    in_file1 = "../videos/hallway.mp4"
    in_file2 = "../videos/hallway.mp4"
    out_file1 = "../detected_videos/out1.mp4"
    out_file2 = "../detected_videos/out2.mp4"
    tracker_thread1 = threading.Thread(target=save_tracker, args=(in_file1, out_file1, model1, (480, 848), 30),
                                       daemon=True)
    tracker_thread2 = threading.Thread(target=save_tracker, args=(in_file2, out_file2, model2, (480, 640), 30),
                                       daemon=True)
    tracker_thread1.start()
    tracker_thread2.start()
    tracker_thread1.join()
    tracker_thread2.join()


if __name__ == '__main__':
    main()
