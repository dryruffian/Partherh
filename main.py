import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue

# Global variables
roi_points = []
temp_point = None
dragging = False
processing = False
density_queue = Queue()
frame_queue = Queue(maxsize=1)  # For storing the latest frame

def load_yolo():
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print("YOLO model loaded successfully")
        return net, output_layers, classes
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        raise

def detect_vehicles(frame, net, output_layers, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    car_count, bike_count = 0, 0
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                if classes[class_id] == "car":
                    car_count += 1
                elif classes[class_id] in ["motorcycle", "bicycle"]:
                    bike_count += 1

    return car_count, bike_count, boxes, confidences, class_ids

def calculate_density(frame, roi_mask, net, output_layers, classes):
    roi_area = cv2.countNonZero(roi_mask) / 10000  # Assume 100 pixels = 1 meter
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    car_count, bike_count, boxes, confidences, class_ids = detect_vehicles(masked_frame, net, output_layers, classes)
    car_density = car_count / roi_area
    bike_density = bike_count / roi_area
    return car_density, bike_density, boxes, confidences, class_ids

def process_video_thread(video_path, roi_mask, roi_points):
    global processing
    processing = True
    print("Video processing thread started")
    cap = cv2.VideoCapture(video_path)
    net, output_layers, classes = load_yolo()

    frame_count = 0
    while processing:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        frame_count += 1
        if frame_count % 5 == 0:  # Process every 5th frame for better performance
            # Calculate density and get bounding boxes
            car_density, bike_density, boxes, confidences, class_ids = calculate_density(frame, roi_mask, net, output_layers, classes)
            density_queue.put((car_density, bike_density))

            # Draw ROI on frame
            frame_with_roi = frame.copy()
            roi_poly = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(frame_with_roi, [roi_poly], (0, 255, 0, 64))
            frame_with_roi = cv2.addWeighted(frame, 0.7, frame_with_roi, 0.3, 0)

            # Draw bounding boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = (0, 1, 0) if label == "car" else (1, 0, 0)  # Green for cars, Red for bikes
                    plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
                    plt.text(x, y - 5, label, fontsize=8, color=color)

            # Convert BGR to RGB for Matplotlib
            frame_rgb = cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB)

            # Put latest frame in queue
            if frame_queue.full():
                frame_queue.get_nowait()  # Remove old frame
            frame_queue.put(frame_rgb)
            print(f"Processed frame {frame_count}")

    cap.release()
    print("Video processing thread ended")

def draw_roi(event):
    global roi_points, temp_point, dragging

    if event.button == 1 and event.name == 'button_press_event':  # Left mouse button pressed
        for i, point in enumerate(roi_points):
            if np.sqrt((event.xdata - point[0])**2 + (event.ydata - point[1])**2) < 10:
                dragging = True
                temp_point = (i, event.xdata, event.ydata)
                return
        roi_points.append((event.xdata, event.ydata))
        temp_point = (event.xdata, event.ydata)

    elif event.name == 'motion_notify_event':  # Mouse moving
        if dragging and temp_point:
            roi_points[temp_point[0]] = (event.xdata, event.ydata)
        else:
            temp_point = (event.xdata, event.ydata)

    elif event.button == 1 and event.name == 'button_release_event':  # Left mouse button released
        dragging = False
        if temp_point and isinstance(temp_point, tuple) and len(temp_point) == 3:
            roi_points[temp_point[0]] = (event.xdata, event.ydata)
        temp_point = None

    plt.clf()
    plt.imshow(first_frame)
    if len(roi_points) > 1:
        roi_array = np.array(roi_points)
        plt.plot(roi_array[:, 0], roi_array[:, 1], 'g-')
        for point in roi_points:
            plt.plot(point[0], point[1], 'ro')
    plt.draw()

def create_roi_mask(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if len(roi_points) > 2:
        roi_poly = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [roi_poly], 255)
    return mask

def process_video(video):
    global roi_points, processing, temp_point, first_frame
    roi_points = []
    temp_point = None

    try:
        print("Opening video file...")
        video_path = video if isinstance(video, str) else video.name
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file", gr.update(visible=False)

        ret, first_frame = cap.read()
        if not ret:
            return "Error: Could not read first frame", gr.update(visible=False)

        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        print("Starting ROI selection...")
        fig, ax = plt.subplots()
        ax.imshow(first_frame)
        ax.set_title("Define ROI: Click to add points, press 'c' to confirm")
        fig.canvas.mpl_connect('button_press_event', draw_roi)
        fig.canvas.mpl_connect('motion_notify_event', draw_roi)
        fig.canvas.mpl_connect('button_release_event', draw_roi)
        plt.show()

        if len(roi_points) < 3:
            return "Not enough points to define ROI", gr.update(visible=False)

        print("Creating ROI mask...")
        roi_mask = create_roi_mask(first_frame)

        print("Starting video processing thread...")
        # Start processing thread
        Thread(target=process_video_thread, args=(video_path, roi_mask, roi_points), daemon=True).start()

        print("Video processing initiated")
        return gr.update(visible=True), gr.update(visible=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", gr.update(visible=False)

def get_density_and_frame():
    print("Attempting to update frame and density...")
    if not density_queue.empty() and not frame_queue.empty():
        car_density, bike_density = density_queue.get()
        frame = frame_queue.get()

        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.text(10, 30, f"Car Density: {car_density:.2f} vehicles/m²", color='green', fontsize=10)
        ax.text(10, 60, f"Bike Density: {bike_density:.2f} vehicles/m²", color='red', fontsize=10)
        ax.axis('off')

        print("Frame and density updated successfully")
        return fig
    else:
        print("Queue is empty, no update performed")
        return None

def stop_processing():
    global processing
    processing = False
    return "Processing stopped"

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Real-time Vehicle Density Calculator")
    with gr.Row():
        video_input = gr.Video()
        video_output = gr.Plot(label="Processed Video", visible=False)
    with gr.Row():
        process_btn = gr.Button("Process Video")
        stop_btn = gr.Button("Stop Processing", visible=False)
    status_text = gr.Textbox(label="Status", value="Ready")

    def process_and_update_status(video):
        result = process_video(video)
        if isinstance(result, tuple):
            status = result[0]
            return result[1], gr.update(visible=True), status
        else:
            return gr.update(visible=True), gr.update(visible=True), "Video processing started"

    process_btn.click(
        process_and_update_status,
        inputs=[video_input],
        outputs=[video_output, stop_btn, status_text]
    )
    stop_btn.click(stop_processing, outputs=[status_text])
    video_output.change(get_density_and_frame, None, video_output, every=1)

if __name__ == "__main__":
    iface.launch()