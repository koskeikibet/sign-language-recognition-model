import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy for numerical operations

background = None  # Initialize background to None for capturing static background
accumulated_weight = 0.5  # Weight for averaging in the background

ROI_top = 100  # Define top boundary for Region of Interest (ROI)
ROI_bottom = 300  # Define bottom boundary for ROI
ROI_right = 150  # Define right boundary for ROI
ROI_left = 350  # Define left boundary for ROI


def cal_accum_avg(frame, accumulated_weight):  # Function to calculate accumulated average for background
    global background  # Use global variable for background

    if background is None:  # If background is not set
        background = frame.copy().astype("float")  # Copy the frame to background and convert to float type
        return None  # Exit the function after setting the background

    cv2.accumulateWeighted(frame, background, accumulated_weight)  # Update background with weighted average


def segment_hand(frame, threshold=25):  # Function to segment the hand from the background
    global background  # Use global variable for background

    diff = cv2.absdiff(background.astype("uint8"), frame)  # Calculate absolute difference between background and frame
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)  # Apply threshold to get binary image

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    if len(contours) == 0:  # If no contours found
        return None  # Return None if no hand is detected
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)  # Get the largest contour by area
        return (thresholded, hand_segment_max_cont)  # Return thresholded image and largest contour


cam = cv2.VideoCapture(0)  # Start video capture from the default camera

num_frames = 0  # Initialize frame counter
element = 10  # Set the element or gesture number to be detected
num_imgs_taken = 0  # Initialize counter for images taken

while True:  # Start an infinite loop for real-time processing
    ret, frame = cam.read()  # Read a frame from the camera

    frame = cv2.flip(frame, 1)  # Flip the frame to avoid inverted image
    frame_copy = frame.copy()  # Make a copy of the frame

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]  # Extract the Region of Interest (ROI)

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert ROI to grayscale
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)  # Apply Gaussian blur for noise reduction

    if num_frames < 60:  # For the first 60 frames
        cal_accum_avg(gray_frame, accumulated_weight)  # Update background with accumulated average
        if num_frames <= 59:  # If still initializing background
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display wait message
    elif num_frames <= 300:  # After background is set, start segmenting the hand
        hand = segment_hand(gray_frame)  # Segment the hand from the background

        cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display adjustment message

        if hand is not None:  # If hand is detected
            thresholded, hand_segment = hand  # Unpack thresholded image and hand contour

            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)  # Draw contour on hand
            cv2.putText(frame_copy, str(num_frames) + " For " + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display frame count and gesture

            cv2.imshow("Thresholded Hand Image", thresholded)  # Show the thresholded hand image

    else:  # After 300 frames, start capturing images
        hand = segment_hand(gray_frame)  # Segment hand from background

        if hand is not None:  # If hand is detected
            thresholded, hand_segment = hand  # Unpack thresholded image and hand contour

            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)  # Draw contour on hand
            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display frame count
            cv2.putText(frame_copy, str(num_imgs_taken) + " images for " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display image count

            cv2.imshow("Thresholded Hand Image", thresholded)  # Show the thresholded hand image

            if num_imgs_taken <= 300:  # If fewer than 300 images have been taken
                cv2.imwrite(r"D:\\gesture\\x\\" + str(num_imgs_taken) + '.jpg', thresholded)  # Save the thresholded image
            else:
                break  # Exit loop if 300 images are taken
            num_imgs_taken += 1  # Increment image count
        else:
            cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display message if no hand detected

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)  # Draw the ROI rectangle on the frame

    cv2.putText(frame_copy, "sign language recognition system", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)  # Display application title

    num_frames += 1  # Increment the frame count

    cv2.imshow("Sign Detection", frame_copy)  # Display the frame with hand detection

    k = cv2.waitKey(1) & 0xFF  # Wait for keypress
    if k == 27:  # Exit if 'Esc' key is pressed
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
cam.release()  # Release the camera
