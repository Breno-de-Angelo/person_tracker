from ultralytics import YOLO
import numpy as np
import cv2


def save_tracker(in_file, out_file, model, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI or 'mp4v' for MP4
    out = cv2.VideoWriter(out_file, fourcc, fps, frame_size)
    last_people_detection = []
    count = 0
    draw_door = np.array([[355, 10], [365, 12], [373, 213], [366, 234]], dtype=np.int32)
    # draw_door = np.array([[307, 2], [329, 7], [344, 298], [328, 348]], dtype=np.int32)
    door = draw_door.reshape((-1, 1, 2))
    for result in model.track(source=in_file, stream=True, classes=0):  # for each frame
        current_people_detection = []
        print(f"last_people_detection: {last_people_detection}")
        for box, cls in zip(result.boxes.xywh, result.boxes.cls):  # for each detected object
            if result.boxes.id is not None:
                for detected_id in result.boxes.id:  # for each id detected
                    current_people_detection.append(detected_id)
                    for person in last_people_detection:
                        if person[0] == detected_id:  # person has already been seen
                            person[1] = box  # update position of the person
                            break
                    else:
                        # new person identified
                        print("new person identified")
                        last_people_detection.append([detected_id, box])
                        centroid = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
                        # check if the person appeared in the region of a door
                        dist = cv2.pointPolygonTest(door, centroid, measureDist=True)
                        print(f"centroid = {centroid}")
                        print(f"dist = {dist}")
                        print(f"width = {box[2]}")
                        if dist > -(box[2] * 1.1):  # person close enough to the door
                            # print('-1')
                            count = count - 1
                        print(f"count = {count}")
        print(f"updated last_people_detection: {last_people_detection}")
        print(f"current_people_detection: {current_people_detection}")
        print("")
        for person in last_people_detection:  # for each person previously ou currently detected
            if person[0] not in current_people_detection:  # if the person was not currently detected
                # person disappeared
                print("person disappeared")
                last_people_detection.remove(person)
                box = person[1]
                centroid = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
                # check if the person disappeared in the region of a door
                dist = cv2.pointPolygonTest(door, centroid, measureDist=True)
                print(f"dist = {dist}")
                if dist > -(box[2] * 1.1):  # person close enough to the door
                    # print('+1')
                    count = count + 1
                print(f"count = {count}")
        image_drawn = cv2.polylines(img=result.plot(), pts=[draw_door], isClosed=True, color=(0, 0, 255), thickness=2)
        image_drawn = cv2.putText(img=image_drawn, text=str(count), org=(360,90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=1, color=(0, 255, 0), thickness=2)
        out.write(image_drawn)
    out.release()


def main():
    model = YOLO("../models/yolov8n.pt")
    in_file = "../videos/hallway.mp4"
    out_file = "../detected_videos/hallway_detected.mp4"
    save_tracker(in_file, out_file, model, (848, 480), 30)


if __name__ == '__main__':
    main()
