import sys
import os
from pathlib import Path
import yaml
from clearml.automation import PipelineController
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory

"""
VLM model end-to-end MLOps pipeline. The pipeline is designed to cater for flexibilities of different needs at 
different point of the pipeline. 

Sometimes user may want to execute a task for specific purposes and then continue with the pipeline. Some steps can 
be skipped if the minimum parameters are not provided as specified in the pipeline parameter descriptions (refer to 
the corresponding tasks for more info).

Pipeline parameter settings can be set in each new run for various purposes as follow:

1. End-to-end from downloading base dataset from remote URL to model publising.
2. Skip step 1 URL download, use existing base datatset, and start from step 2: dataset processing
3. Skip steps 1 & 2, use processed dataset and start from step 3: model training
4. Skip steps 1, 2, & 3, use existing model as the new model for evaluation, starts from step 4: model evaluation

The above scenarios are designed to reduce execution time, resources requirements, duplications, and allows adjustment 
to various circumstances. Note that you can not skip model evaluation - this leads to publishing the model. If this is 
not a desire behaviour, use the task Model Evaluation from the WebUI instead of the pipeline.

IMPORTANT: by default, it will use the base_dataset and eval_dataset existing on the server, presuming they are already 
uploaded. If those datasets are not uploaded, please put in the base_dataset_url and/or eval_dataset_url accordingly.
Alternatively, before running the pipeline with default settings, upload the dataset using the following tasks from
the ClearML WebUI:

Upload Base Dataset - upload base dataset. This will trigger default pipeline to run in CD phase (NOT IMPLEMENTED)
Upload Evaluation Dataset - upload base dataset. This will trigger default pipeline to run in CD phase (NOT IMPLEMENTED)
"""
import os
os.chdir("/content/AIS_Project/image_description")

# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')
pipeline_name = "VLMPipeline"

# Connecting ClearML with the current pipeline, from here on everything is logged automatically
pipe = PipelineController(name=pipeline_name, 
                          project=project_name, 
                          add_pipeline_tags=False)
pipe.set_default_execution_queue("desc_preparation")
#pipe._task.set_script(working_dir="/content/AIS_Project/image_description")
""" 
STEP 1: Create Image-Label Mapping dataset from Base dataset under Detection Project
"""
# intial dataset to download. If none provided, task will complete without upload
base_dataset_id = ""
base_dataset_name = "base_dataset_zip"

pipe.add_parameter("base_dataset_id", base_dataset_id, "latest of base_dataset_zip id")
pipe.add_parameter("base_dataset_name", base_dataset_name, "latest of base_dataset_zip name")
def pre_base_dataprep_callback(pipeline, node, param_override) -> bool:    
    print("Cloning step1_desc_basedata_preparation id={}".format(node.base_task_id))    
    return True
def post_base_dataprep_callback(pipeline, node) -> None:   
    print("Completed step1_desc_basedata_preparation id={} {}".format(node.base_task_id, node.executed))    
    return
pipe.add_step(
    name="BaseData_Mapping",
    base_task_project=project_name,
    base_task_name="step1_desc_basedata_preparation",
    parameter_override={
        "General/base_dataset_id": "${pipeline.base_dataset_id}",
        "General/base_dataset_name": "${pipeline.base_dataset_name}"
        },
    pre_execute_callback=pre_base_dataprep_callback,
    post_execute_callback=post_base_dataprep_callback
)
""" 
STEP 2: Create Image-Label Mapping dataset from Eval dataset under Detection Project
"""
eval_dataset_id = ""
eval_dataset_name = "eval_dataset_zip"

pipe.add_parameter("eval_dataset_id", eval_dataset_id, "latest of eval_dataset_zip id")
pipe.add_parameter("eval_dataset_name", eval_dataset_name, "latest of eval_dataset_zip name")

def pre_base_dataprep_callback(pipeline, node, param_override) -> bool:    
    print("Cloning step2_desc_testdata_preparation id={}".format(node.base_task_id))    
    return True
def post_base_dataprep_callback(pipeline, node) -> None:   
    print("Completed step2_desc_testdata_preparation id={} {}".format(node.base_task_id, node.executed))    
    return
pipe.add_step(
    name="EvalData_Mapping",
    base_task_project=project_name,
    base_task_name="step2_desc_testdata_preparation",
    parameter_override={
        "General/eval_dataset_id": "${pipeline.eval_dataset_id}",
        "General/eval_dataset_name": "${pipeline.eval_dataset_name}"
        },
    pre_execute_callback=pre_base_dataprep_callback,
    post_execute_callback=post_base_dataprep_callback
)

remote_execution = project.get("pipeline-remote-execution")
if remote_execution:
    print(f"Executing '{pipeline_name}' pipeline remotely")
    pipe.start(queue = "desc_preparation")
else:
    print(f"Executing '{pipeline_name}' pipeline locally")
    pipe.start_locally(run_pipeline_steps_locally=True)
print("done")