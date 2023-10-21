import cv2
import tensorflow as tf
import numpy as np
import time  # Import the time library

# Load the TensorFlow model
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
model_path = './ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_path)

# Function to draw the rectangle around each detected person and collect coordinates
def draw_rectangles_and_collect_coords(frame, boxes, scores, classes):
    coords = []  # List to store coordinates of people
    height, width, _ = frame.shape
    for i in range(len(boxes)):
        if scores[i] > 0.5 and classes[i] == 1:
            box = boxes[i]
            if isinstance(box, np.ndarray) and box.shape[0] == 4:
                y1, x1, y2, x2 = box
                x1 *= width
                x2 *= width
                y1 *= height
                y2 *= height
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                coords.append((center_x, center_y))  # Save the coordinates
            else:
                print(f"Unexpected box format at index {i}: {box}")
    return frame, coords  # Return both frame and coordinates

def main():
    cap = cv2.VideoCapture(1)
    last_time = 0  # Initialize a variable to store the last time coordinates were printed

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame")
            break

        # Preprocess the image for the model
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = cv2.resize(input_frame, (320, 320))
        input_tensor = tf.convert_to_tensor(input_frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform the detection
        detections = model(input_tensor)

        # Extract detection results
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        # Draw rectangles around detected persons
        frame_with_detections, coords = draw_rectangles_and_collect_coords(frame, boxes, scores, classes)

        # Check if at least 1 second has passed since the last print
        if time.time() - last_time >= 1:
            for i, coord in enumerate(coords, 1):
                print(f"people-{i}: {coord[0]}, {coord[1]}")
            last_time = time.time()  # Update the last time to the current time

        cv2.imshow('Frame', frame_with_detections)

        # Break the loop if "q" is pressed
        # Press 'q' in the camera frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()