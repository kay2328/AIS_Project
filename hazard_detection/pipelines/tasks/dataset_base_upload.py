import argparse
import os
from clearml import Task, Dataset, StorageManager

"""
Upload zipped YOLO dataset file from remote URL, extract and upload to ClearML server. 
The zipped file needs to contain the YAML file and assets in the following structure:

data.yaml
images/
labels/
"""

task = Task.init(project_name="Hazard Detection", 
                task_name="Upload Base Dataset", 
                task_type=Task.TaskTypes.data_processing,
                reuse_last_task_id=True)

params = {
    'dataset_url': 'https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip'
}

task.connect(params)
task.execute_remotely(queue_name="default")

dataset_url = params['dataset_url']

if dataset_url:
    hazard_dataset = StorageManager.get_local_copy(remote_url=dataset_url,
                                                   name="base_dataset",
                                                   cache_context="hd_zip_dataset",
                                                   force_download=True)

    print("Downloading to: ", hazard_dataset)

    if hazard_dataset is None:
        # Error: Assume file not found (404 http status code)
        raise FileNotFoundError("404", f"Found not found at URL {dataset_url}") 

    dataset = Dataset.create(
        dataset_project="Hazard Detection", dataset_name="base_dataset"
    )

    dataset.add_files(path=hazard_dataset)

    print('Uploading dataset in the background')

    dataset.upload()
    dataset.finalize()