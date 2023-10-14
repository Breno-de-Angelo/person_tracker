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
            vertices.append([x, y])

    # Create a window and set the mouse callback function
    cv2.namedWindow('Video Frame')
    cv2.setMouseCallback('Video Frame', get_coordinates)

    # Display the first frame and wait for a user click
    cv2.imshow('Video Frame', frame)
    cv2.waitKey(0)

    # Clean up and release the video capture
    cv2.destroyAllWindows()
    cap.release()

    return vertices
