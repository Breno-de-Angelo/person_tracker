from ultralytics import YOLO
import numpy as np
import cv2
from get_xy_position_from_video import get_xy_position_from_video
import logging
import argparse


def my_tracker(in_file, model, file_index, door_vertices):
    video = cv2.VideoCapture(in_file)  # Read the video file
    last_people_detection = []
    number_of_doors = int(len(door_vertices) / 4)
    draw_door = []
    door = []
    count = [0 for _ in range(number_of_doors)]
    text_position = []
    logging.debug(door_vertices)
    for i in range(number_of_doors):
        draw_door.append(np.array(door_vertices[(4*i):(4*(i + 1))], dtype=np.int32))
    for item in draw_door:
        door.append(item.reshape((-1, 1, 2)))
    logging.debug(draw_door)
    for item in draw_door:
        max_x = -1000000
        min_x = 1000000
        max_y = -1000000
        min_y = 1000000
        for i in range(4):
            if item[i][0] > max_x:
                max_x = item[i][0]
            if item[i][0] < min_x:
                min_x = item[i][0]
            if item[i][1] > max_y:
                max_y = item[i][1]
            if item[i][1] < min_y:
                min_y = item[i][1]
        text_position.append((int((min_x + max_x) / 2), int((min_y + max_y) / 2)))

    while True:
        ret, frame = video.read()  # Read the video frames
        if not ret:  # Exit the loop if no more frames
            break
        result = model.track(frame, persist=True, classes=0)[0]
        current_people_detection = []
        logging.debug(f"last_people_detection: {last_people_detection}")
        for box, cls in zip(result.boxes.xywh, result.boxes.cls):  # for each detected object
            if result.boxes.id is not None:
                for detected_id in result.boxes.id:  # for each id detected
                    current_people_detection.append(detected_id)
                    for person in last_people_detection:
                        if person[0] == detected_id:  # person has already been seen
                            person[1] = box  # update position of the person
                            break
                    else:
                        logging.debug("new person identified")
                        last_people_detection.append([detected_id, box])
                        centroid = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
                        # check if the person appeared in the region of a door
                        dist = float('-inf')
                        door_id = 1000000
                        for i in range(number_of_doors):
                            calculated_dist = cv2.pointPolygonTest(door[i], centroid, measureDist=True)
                            if calculated_dist > dist:
                                dist = calculated_dist
                                door_id = i
                        logging.debug(f"centroid = {centroid}")
                        logging.debug(f"dist = {dist}")
                        logging.debug(f"width = {box[2]}")
                        logging.debug(f"door_id = {door_id}")
                        if dist > -(box[2] * 1.1):  # person close enough to the door
                            count[door_id] = count[door_id] - 1
                        logging.debug(f"count = {count[door_id]}")
        logging.debug(f"updated last_people_detection: {last_people_detection}")
        logging.debug(f"current_people_detection: {current_people_detection}")
        logging.debug("")
        for person in last_people_detection:  # for each person previously ou currently detected
            if person[0] not in current_people_detection:  # if the person was not currently detected
                # person disappeared
                logging.debug("person disappeared")
                last_people_detection.remove(person)
                box = person[1]
                centroid = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
                # check if the person disappeared in the region of a door
                dist = float('-inf')
                door_id = 1000000
                for i in range(number_of_doors):
                    calculated_dist = cv2.pointPolygonTest(door[i], centroid, measureDist=True)
                    if calculated_dist > dist:
                        dist = calculated_dist
                        door_id = i
                if dist > -(box[2] * 1.1):  # person close enough to the door
                    # print('+1')
                    count[door_id] = count[door_id] + 1
                logging.debug(f"dist = {dist}")
                logging.debug(f"count = {count[door_id]}")
        image_drawn = result.plot()
        for i in range(number_of_doors):
            image_drawn = cv2.polylines(img=image_drawn, pts=[draw_door[i]], isClosed=True,
                                        color=(255, 0, 0), thickness=2)
            image_drawn = cv2.putText(img=image_drawn, text=str(count[i]), org=text_position[i],
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow(f"Tracking_Stream_{file_index}", image_drawn)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    # Release video sources
    video.release()


def main(input_file, model_path):
    logging.basicConfig(level=logging.ERROR)
    model = YOLO(model_path)
    if input_file.isnumeric():
        input_file = int(input_file)
        input_file = int(input_file)
    while True:
        door_vertices = get_xy_position_from_video(input_file)
        if len(door_vertices) % 4 == 0:  # Assert a valid number of vertices
            break
        logging.error("Select a number of vertices multiple of 4")
    my_tracker(input_file, model, 1, door_vertices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to count the number of people inside each room using 1 camera")

    parser.add_argument('-input_file', required=True, help="Input file path")
    parser.add_argument('-model', required=True, help="Model path")

    args = parser.parse_args()

    main(args.input_file, args.model)
