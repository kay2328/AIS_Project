
# Image Description 
This component will perform image description using VLM as part of the system. THis project will acquire dataset, train the models and produce the final model for production using MLOps pipeline.

## Datasets:
### Image description/caption generation:
The dataset to be utilized for this purpose is the VizWiz-Captions dataset, which consists of 39,181 images obtained from people who are blind and each of them are paired with 5 captions. The distribution of the dataset is as follows:
1. Train set:
   23,431 images and 117,155 captions
2. Validation:
   7,750 images and 38,750 captions
3. Test set:
   8,000 images and 40,000 captions

Link to the dataset source: https://vizwiz.org/tasks-and-datasets/image-captioning

## Notebooks
Jupyter notebooks used for EDA, model testing and evaluation, model training, and deployment.

## Models 
Final trained model goes into this folder. Updated model will be used to deploy into the API and/or UI component.

## Setup