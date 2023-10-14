import cv2

def get_xy_position_from_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Read the first frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    vertices = []

    # Create a callback function to capture mouse events
    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
            if len(vertices) % 5 != 4 or len(vertices) == 0:
                vertices.append([x, y])
            else:
                if abs(x - vertices[-4][0]) < 30 and abs(y - vertices[-4][1]) < 30:
                    vertices.append(vertices[-4])
        if event == cv2.EVENT_MOUSEMOVE:  # Update the cursor coordinates
            param['cursor_x'], param['cursor_y'] = x, y

    # Create a window and set the mouse callback function
    cv2.namedWindow('Video Frame')
    cursor_x, cursor_y = 0, 0
    cursor_params = {'cursor_x': cursor_x, 'cursor_y': cursor_y}
    cv2.setMouseCallback('Video Frame', get_coordinates, cursor_params)

    original_frame = frame.copy()
    # Main loop
    while True:
        frame = original_frame.copy()
        if len(vertices) != 0:
            for vertex in vertices:  # Draw gray circles at vertices
                cv2.circle(frame, center=vertex, radius=5, color=(100, 100, 100), thickness=-1)
            number_of_doors = len(vertices) // 5
            for i in range(number_of_doors):  # Draw complete doors
                for j in range(5*i+1, 5*(i+1)):
                    cv2.line(frame, pt1=vertices[j-1], pt2=vertices[j], color=(255, 0, 0), thickness=2)
            if len(vertices) > 0:
                for i in range(5*number_of_doors+1, len(vertices)):  # Draw incomplete doors
                    cv2.line(frame, pt1=vertices[i-1], pt2=vertices[i], color=(255, 0, 0), thickness=2)
            # for i in range(1, len(vertices)):
            #     cv2.line(frame, pt1=vertices[i-1], pt2=vertices[i], color=(255, 0, 0), thickness=2)
            # Draw a line from the center point to the cursor
            if len(vertices) % 5 != 0:
                cv2.line(frame, pt1=vertices[-1], pt2=(cursor_params['cursor_x'], cursor_params['cursor_y']),
                         color=(0, 0, 255), thickness=2)

        # Show the frame
        cv2.imshow('Video Frame', frame)

        key = cv2.waitKey(1)
        # Check for 'Esc' key to undo
        if key == 27:
            if len(vertices) != 0:
                vertices.pop()
        # Check for 'Enter' to exit
        elif key == 13:
            break

    # Clean up and release the video capture
    cv2.destroyAllWindows()
    cap.release()

    return [vertices[i] for i in range(len(vertices)) if i % 5 != 0]