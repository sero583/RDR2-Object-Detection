import threading
import os
import argparse
from datetime import datetime
import cv2
import numpy as np
import pygetwindow as gw
import screeninfo
import pyautogui
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from roboflow import Roboflow

# Sample video accessable here: https://youtu.be/185HsB_F2cs
# Any other gameplay footage can also be used, however it's ideal to have the HUD (things such as map, health bar, etc.) to disabled. That is the case in my sample video.

# Create argument parser
parser = argparse.ArgumentParser(description="Red Dead Redemption 2 Object Detection using YOLOv8 as a model.")
# Add an argument, which only makes this script train
parser.add_argument("--train-only", action="store_true", help="Only generates training files, instead of real-time screen casting.")
# Add an argument to be able to specify epochs
parser.add_argument("--epochs", default=50, help="Set the epcoh amount.")
# Add an argument for dataset version specifying
parser.add_argument("--dataset-version", default=3, help="Choose which dataset version should be used from roboflow.")
# Add an argument to supply roboflow API key
parser.add_argument("--rf-api-key", default=None, help="Supply API key for roboflow.")
# Add an argument to supply roboflow workspace
parser.add_argument("--rf-workspace", default=None, help="Supply workspace name from roboflow.")
# Add an argument to supply roboflow project
parser.add_argument("--rf-project", default=None, help="Supply project name from roboflow.")
# Add an argument for recording
parser.add_argument("--record", action="store_true", help="Enables recording the footage onto a file in the folder ./recordings/")
# Add an argument for fps of recording
parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60)")
# Add an argument for the screenshot interval
parser.add_argument("--interval", type=int, default=25, help="Interval between frames in milliseconds (default: 25)")
# Add an argument to ignore if game is not active
parser.add_argument("--ignore-game-not-active", action="store_true", help="Skip the verification if the game window is active or not")

# Parse the command-line arguments
args = parser.parse_args()

# Create datasets folder, except it's available already
os.makedirs("./datasets/", exist_ok=True)
# Dataset path
DATASET_DATA_YAML_PATH = os.path.join(os.getcwd(), "datasets/RDR2-Object-Detection-" + str(args.dataset_version) + "/data.yaml")

# build new model
model = YOLO("yolov8n.pt")
# train the model, adjust to current used dataset version, to get a download copy go into datasets and run download_dataset.py
model.train(data=DATASET_DATA_YAML_PATH, epochs=50)

if args.train_only:
    print("Starting training process with " + str(args.epochs) + " epochs...")
    model.train(data=DATASET_DATA_YAML_PATH, epochs=args.epochs)
    print("Training finished, exiting script...")
    exit()

# Check if dataset version is on machine
if not os.path.exists(DATASET_DATA_YAML_PATH):
    # Assuming dataset isn't downloaded yet, download the desired version.
    # Just supply unchecked arguments, since roboflow will simply throw an exception if theyre invalid.
    print("Trying to download dataset using supplied credentials...")
    rf = Roboflow(api_key=args.rf_api_key)
    project = rf.workspace(args.rf_workspace).project(args.rf_project)
    dataset = project.version(args.dataset_version).download("yolov8", output_directory="./datasets/")
else: print("Dataset found, using version " + str(args.dataset_version))

# Verify if the game window is active
if not args.ignore_game_not_active:
    window_title = "Red Dead Redemption 2"
    game_windows = gw.getWindowsWithTitle(window_title)

    if not game_windows:
        print(f"Error: Window with title \"{window_title}\" not found. Exiting program.")
        exit()

    game_window = game_windows[0]
    x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
else: # Just use main monitor - since pygetwindow has no multi-monitor support just use resolution of monitor
    # Get info of main monitor
    main_monitor = screeninfo.get_monitors()[0]
    # Set it"s data to the used variables
    x, y, width, height = main_monitor.x, main_monitor.y, main_monitor.width, main_monitor.height
    

# Load annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Function to update the displayed image in the animation
def update(frame):
    # Capture the screen image within the ROI
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Perform object detection using the model
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    # Create labels
    labels = [
        f"{model.model.names[class_id]}: {confidence:.2%}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # Draw bounding boxes
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame,
        detections=detections
    )
    # Label detections
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    # Update the displayed image
    img_plot.set_array(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    return img_plot,


# Function to handle key presses
def on_key(event):
    if event.key == "q" or event.key == "escape":
        # Close window
        plt.close()
        # Allow any remaining processes to complete before exiting
        plt.pause(2)
        animation.event_source.stop()
        # Join the threads before exiting
        recording_thread.join()

        # Finally shutdown script
        exit()

# Initialize GUI
fig, ax = plt.subplots()
# Turn off axis for cleaner display
ax.axis("off")
# Set a custom title for the window
fig.canvas.manager.set_window_title("Red Dead Redemption 2 Object Detection using YOLOv8")

# Adjust subplot parameters to remove white bars
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Initialize an empty image plot
img_plot = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

# Start the animation
animation = FuncAnimation(fig, update, blit=True, interval=args.interval)

# Function to run the recording
def run_recording():
    if args.record:
        print("Recording active.")
        # Check if folder exists, if not create
        os.makedirs("./recordings/", exist_ok=True)
        
        filename = "./recordings/" + datetime.now().strftime("%Y-%m-%dT%H_%M_%S") + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(filename, fourcc, args.fps, (width, height))

        try:
            while getattr(animation, "running", True):
                animation._draw_next_frame()
                frame = np.frombuffer(animation._encoded_content, dtype=np.uint8)
                frame = frame.reshape(height, width, 4)[:, :, :3]
                out.write(frame)
        finally:
            out.release()


# Create threads for recording
recording_thread = threading.Thread(target=run_recording)
# Start the recording thread
recording_thread.start()

# Show window
plt.show()
