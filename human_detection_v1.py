import cv2
import tensorflow as tf
import numpy as np

# Load the TensorFlow model
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
model_path = './ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_path)

# Function to draw the rectangle around each detected person
def draw_rectangles(frame, boxes, scores, classes):
    height, width, _ = frame.shape
    for i in range(len(boxes)):  # Iterating over the number of boxes
        if scores[i] > 0.5 and classes[i] == 1:  # Class 1 represents human in COCO dataset, adjust if necessary
            # Correcting the box extraction. The boxes variable should be a 2D array.
            box = boxes[i]
            # Ensure box is an array with the coordinates
            if isinstance(box, np.ndarray) and box.shape[0] == 4:
                y1, x1, y2, x2 = box
                x1 *= width
                x2 *= width
                y1 *= height
                y2 *= height
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                print(f"Unexpected box format at index {i}: {box}")
    return frame

def main():
    cap = cv2.VideoCapture(1)  # Use 1 or the index of your camera

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame")
            break

        # Preprocess the image for the model
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = cv2.resize(input_frame, (320, 320))  # Adjust the size based on the model's requirement
        input_tensor = tf.convert_to_tensor(input_frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform the detection
        detections = model(input_tensor)

        # Extract detection results
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        # Draw rectangles around detected persons
        frame_with_detections = draw_rectangles(frame, boxes, scores, classes)

        # Display the frame
        cv2.imshow('Frame', frame_with_detections)

        # Break the loop if "q" is pressed
        # Press 'q' in the camera frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any window once the application exits
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()