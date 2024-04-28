# Import necessary libraries
import os
import cvzone
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Open webcam video capture
cap = cv2.VideoCapture(0)
# Set resolution of the video capture
cap.set(3, 640)  # Width
cap.set(4, 800)  # Height
# Set frames per second
cap.set(cv2.CAP_PROP_FPS, 60)
# Initialize SelfiSegmentation module for background removal
segmentor = SelfiSegmentation()
# Initialize FPS counter
fpsReader = cvzone.FPS()

# Read and resize the background image
imgBg = cv2.imread("images/pexels-quang-nguyen-vinh-222549-6710902.jpg")
imgBg = cv2.resize(imgBg, (640, 480))

# Get a list of file names in the "images" directory
listImg = os.listdir("images")
# Initialize an empty list to store resized images
imgList = []
# Loop through each file in the list
for imgPath in listImg:
    # Read the image
    img = cv2.imread(f'images/{imgPath}')
    # Resize the image to match input image size
    img = cv2.resize(img, (640, 480))
    # Append the resized image to the list
    imgList.append(img)
# Print the number of images in the list
print(len(imgList))

# Initialize index to control which image to use for background removal
indexImg = 0

# Main loop to capture video and perform background removal
while True:
    # Read a frame from the video capture
    success, img = cap.read()
    # Perform background removal using SelfiSegmentation module
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)
    # Stack the original image and the image with background removed vertically
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    # Update and display FPS counter
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))
    # Print the index of the current image used for background removal
    print(indexImg)

    # Display the stacked images
    cv2.imshow("Image", imgStacked)
    # Wait for a key press
    key = cv2.waitKey(1)
    # Control to switch to the previous image in the list
    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
    # Control to switch to the next image in the list
    elif key == ord('d'):
        if indexImg < len(imgList) - 1:
            indexImg += 1
    # Control to quit the program
    elif key == ord('q'):
        break
