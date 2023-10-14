import cv2

# Load the image
image = cv2.imread('test.jpg')

# Create a callback function to capture mouse events
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        print(f'Clicked at (x, y): ({x}, {y})')

# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_coordinates)

# Display the image and wait for a user click
cv2.imshow('Image', image)
cv2.waitKey(0)

# Clean up and close the window
cv2.destroyAllWindows()