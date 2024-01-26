# Object Detection in Red Dead Redemption 2 using YOLOv8

This project made by Serhat Güler implements object detection in the game Red Dead Redemption 2 (short: RDR2) from Rockstar Games. In this project, a custom dataset specially for RDR2 has been made, so YOLO can detect more obstacles in the virtual world.

## Dataset on roboflow universe
Gather your own version at roboflow universie to test this project: https://universe.roboflow.com/yolov8-zpkhg/rdr2-object-detection

## Prerequisites and Installation

1. This project requires python to be installed on your computer. Get python [here](https://github.com/python/cpython). After installing python, the command python's package manager ``pip`` comes alongside with python.
2. Run this command to download all required packages: ```pip install -r requirements.txt```
3. Clone my roboflow universe workspace and generate yourself an API Key to generate yourself a version
4. Use the arguments to enter your credentinals into the python script
5. Start project by using ```py rdr2_obj_detection.py``` (and supply arguments if necessary)

With those previous two steps, you have installed everything, which is required to run this project.

## Usage

This project can be started by using simply ```py rdr2_obj_detection.py```. If ``py`` is not found as a command on your system, simply use ``python``. It's the same.

## Customization

To start this project with different behavior, here is a list of all arguments available.

| Argument                  | Type          | Default Value   | Description                                                                 |
|---------------------------|---------------|-----------------|-----------------------------------------------------------------------------|
| --dataset-version         | Integer       | 3               | Specifies the used dataset version                                          |
| --rf-api-key              | String        | None            | Ability to supply your own API key without leaving it permanently in file   |
| --rf-workspace            | String        | None            | Ability to supply your workspace name                                       |
| --rf-project              | String        | None            | Ability to supply your project name                                         |
| --fps                     | Integer       | 60              | Frames per second                                                           |
| --interval                | Integer       | 25              | Interval between frames in milliseconds                                     |
| --dataset                 | String        | yolov8n.pt      | Path to the YOLO model dataset                                              |
| --model_weights           | String        | yolov3.weights  | Path to YOLO model weights                                                  |
| --config_file             | String        | yolov3.cfg      | Path to YOLO model configuration file                                       |
| --class_names_file        | String        | class_names.txt | Path to YOLO class names file                                               |
| --ignore-game-not-active  | Boolean       | False           | Skip the verification if the game window is active or not                   |

Note that you only need to use initally --rf-api-key, --rfworkspace and --rf-project, so the script can download simply the desired dataset version for you from roboflow. You can also simply download it in YOLOv8 format and place the files in ```dataset/RDR2-Object-Detection-<YOUR_DESIRED_VERSION>```. Create the subfolders, if you don't have them already. ```<YOUR_DESIRED_VERSION>``` must match the version youre using in the argument, if you dont want to change something just name it simply "3".

## Author

This project has been made for the lecture Applied Machine Learning with the lecturer Mr. Prof. Dr. Sebastian Leuoth at Hof University by Serhat Güler (matriculation number: 00342920).