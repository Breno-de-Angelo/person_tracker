import cv2

# Open the video file
cap = cv2.VideoCapture('hallway.mp4')

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

# Create a callback function to capture mouse events
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        print(f'Clicked at (x, y): ({x}, {y})')

# Create a window and set the mouse callback function
cv2.namedWindow('Video Frame')
cv2.setMouseCallback('Video Frame', get_coordinates)

# Display the first frame and wait for a user click
cv2.imshow('Video Frame', frame)
cv2.waitKey(0)

# Clean up and release the video capture
cv2.destroyAllWindows()
cap.release()