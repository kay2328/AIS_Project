# SecondSight by EnigmaAI

SecondSight is an application that uses an iPhone to detect hazard objects for the visually impaired individuals. It can also perform scene description without hazards. It is not designed to replace any existing tools and methods but to compliment any existing methods.

The applicaiton requires a live camera input for object detection using YOLO and speech discription using vision lanaguage model. There are two modes:

1. Standalone - Hazard object detection with haptic feedback. No speech description.
2. Plus - With additional speech description upon detection and at request

This project consists 4 components.

## Cloning fom GitHub 
```sh 
git clone https://{github_username}:{github_access_token}@github.com/{github_username}/EnigmaAI.git
   ```

## Libaray dependencies
1. List all top level libraries in `requirements.txt` for top level.
2. Run `pip install -r requirements.txt` to install top level dependencies

## Hazard Detection component
Model pipeline for training and deploy YOLO model for hazard object detection.

Supports functionality for hazard object detection on live camera video.

Model output will be integrated into the application in API or UI (iOS app) components

For more detail, refer to submodules README.md including setup instructions.

### Install libary dependencies
1. List all libraries in `hazard_detection/requirements.txt`.
2. Run `pip install -r hazard_detection/requirements.txt` to install

## Image Description component
Model pipeline for training and deploy VLM model for image description.

Supports functionality for descrbing image from video frame after hazard is detected or at request any time.

Model output will be integrated into the application in API or UI (iOS app) components

For more detail, refer to submodules README.md including setup instructions.

### Install libary dependencies
1. List all libraries in `image_description/requirements.txt`.
2. Run `pip install -r image_description/requirements.txt` to install

## API component
API to serve the model inferencing endpoint for the application. This is a FastAPI project.

### Install libaray dependencies
1. List all libraries in `api/requirements.txt`.
2. Run `pip install -r api/requirements.txt` to install

## iOS component
UI component for iOS platform. This is the final application using all other componenets to serve its functionalities for a blind person. 

Please refer to Requirment and Design documentations for more details.






