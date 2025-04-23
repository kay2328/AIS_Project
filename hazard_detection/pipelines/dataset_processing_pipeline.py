from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name="Data Processing", project="Hazard Detection", version="1.0.1", add_pipeline_tags=False
)

pipe.add_parameter(
        "dataset_url", "https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip"
)

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="upload_zip_dataset",
    base_task_project="Hazard Detection",
    base_task_name="Upload ZIP Dataset",
    parameter_override={"General/dataset_url": "${pipeline.dataset_url}"},
)

# pipe.add_step(
#     name="extract_dataset",
#     parents=["upload_zip_dataset"],
#     base_task_project="Hazard Detection",
#     base_task_name="Extract and Split Dataset",
#     parameter_override={
#         "General/dataset_url": "${stage_data.artifacts.dataset.url}",
#         "General/test_size": 0.25,
#     },
# )


# for debugging purposes use local jobs
# pipe.start_locally()

# Starting the pipeline (in the background)
pipe.start(queue="default")

print("done")