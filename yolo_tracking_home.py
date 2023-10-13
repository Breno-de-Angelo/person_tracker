from ultralytics import YOLO
import numpy as np
import cv2
import torch

def save_tracker(in_file, out_file, model, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI or 'mp4v' for MP4
    out = cv2.VideoWriter(out_file, fourcc, fps, frame_size)
    people_detected_id = []
    draw_door = np.array([[349, 479], [349, 100], [413, 93], [407, 475]], dtype=np.int32)
    door = draw_door.reshape((-1, 1, 2))
    for result in model.track(source=in_file, stream=True):
        for box, cls in zip(result.boxes.xywh, result.boxes.cls):
            if result.boxes.id is not None:
                for detected_id in result.boxes.id:
                    if cls == 0.0:
                        if detected_id not in people_detected_id:
                            # new person identified
                            people_detected_id.append(detected_id)
                            centroid = (int(box[0] + box[2]/2), int(box[1] + box[3]/2))
                            dist = cv2.pointPolygonTest(door, centroid, measureDist=True)
                            if dist < 0:
                                print('count')
        image_drawn = cv2.polylines(img=result.plot(), pts=[draw_door], isClosed=True, color=(0, 0, 255), thickness=2)
        out.write(image_drawn)
    out.release()


def main():
    model = YOLO("yolov8n.pt")
    in_file = "home.mp4"
    out_file = "out1.mp4"
    save_tracker(in_file, out_file, model, (848, 480), 30)

if __name__ == '__main__':
    main()
