"""
VLMPipeline: end-to-end vision‐language MLOps orchestration using ClearML

This pipeline orchestrates the full life cycle of a vision‐language (VLM) image‐captioning model, providing flexible
entry points so you can run only the steps you need:
1. **BaseData_Mapping** (step1)  
   Download or reuse an object‐detection dataset (images + annotations) and convert it into an image→label JSON mapping.
2. **EvalData_Mapping** (step2)  
   Same as BaseData_Mapping, but for your evaluation images.
3. **Base Caption Generation** (step3)  
   Use a “teacher” model to generate pseudo‐captions for each training image (knowledge distillation setup).
4. **Train/Val Split** (step5)  
   Split the pseudo‐captioned dataset into train/validation JSONs and upload as a new ClearML dataset.
5. **Student Model Training** (step6)  
   Train your VisionEncoderDecoder student model on the split dataset, logging metrics (e.g., CIDEr) to ClearML.
6. **Hyperparameter Optimization** (step7)  
   Run a grid search over `num_epochs`, `batch_size`, `lr`, and `weight_decay` (within a time limit). Save only top trials.
7. **Eval Caption Generation** (step4)  
   With the best student model, generate captions on the evaluation images.
8. **Model Evaluation** (step8)  
   Compare generated vs. ground‐truth eval captions (CIDEr, BLEU, ROUGE) and choose the stronger model.
9. **Model Publishing** (step9)  
   Publish the winning model to ClearML for reuse in downstream applications.

### Flexible Execution Modes
Depending on which pipeline parameters supplied, it can skip any of the first three steps and pick up later:
- **Full run**: steps 1→9 
- **Skip raw download**: start at step 2  
- **Skip prep + labeling**: start at step 3  
- **Skip prep + training**: start at step 4 (directly evaluate an existing model)
> **Note:** Model evaluation (step 8) cannot be skipped it would auto‐publish whatever draft model provided.
All steps log their task IDs and parameters to the console, so you can trace exactly how each subtask was cloned
and run. Simply override the pipeline parameters at launch time to adjust behavior, queues, or hyperparameter ranges.
"""

import sys
import os
from clearml.automation import PipelineController
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory
os.chdir("/content/AIS_Project/")

# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')
pipeline_name = "VLM Pipeline"

# Connecting ClearML with the current pipeline, from here on everything is logged automatically
pipe = PipelineController(name=pipeline_name, 
                          project=project_name, 
                          add_pipeline_tags=False)
pipe.set_default_execution_queue(project.get('queue-gpu'))
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

pipe.add_parameter("eval_dataset_id", "", "latest of eval_dataset_zip id")
pipe.add_parameter("eval_dataset_name", "eval_dataset_zip", "latest of eval_dataset_zip name")

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


""" 
STEP 3: Train Data Reference description generation
"""
dataset_id = ""
dataset_name = "Desc_Base_Dataset"
base_dataset_id = ''
base_dataset_name = "base_dataset_zip"

pipe.add_parameter("dataset_id", dataset_id, "latest id of base data img-label mapping")
pipe.add_parameter("dataset_name", dataset_name, "latest of base data img-label name")
pipe.add_parameter("base_dataset_id", base_dataset_id, "latest of base_dataset_zip id")
pipe.add_parameter("base_dataset_name", base_dataset_name, "latest of base_dataset_zip name")

def pre_processing_callback(pipeline, node, param_override) -> bool:
    print("Cloning step3_desc_basecaption_generation id={}".format(node.base_task_id))    
    return True
def post_processing_callback(pipeline, node) -> None:
    print("Completed step3_desc_basecaption_generation id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="base_desc_generation",
    parents=["BaseData_Mapping"],
    base_task_project=project_name,
    base_task_name="step3_desc_basecaption_generation",
    parameter_override={
        "General/dataset_id": "${BaseData_Mapping.parameters.General/output_dataset_id}",
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/base_dataset_id": "${pipeline.base_dataset_id}", 
        "General/base_dataset_name": "${pipeline.base_dataset_name}"
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
)

""" 
STEP 4: Splitting Train Dataset
"""
# it will get dataset_id from step 3, if not provided, this will be used
cap_dataset_id= ''
cap_dataset_name= 'Desc_Caption_BaseDataset'
random_state= 42
val_size=0.15
pipe.add_parameter("cap_dataset_id", "", "(Optional) Overitten if previous task is not skipped. If empty, use the latest of base caption dataset id")
pipe.add_parameter("cap_dataset_name", "Desc_Caption_BaseDataset", "latest of base caption dataset_name")
pipe.add_parameter("random_state", 42, "Specify random state for consistent training")
pipe.add_parameter("val_size", 0.15, "Validation split. Percentage of entire dataset.")
pipe.add_parameter("split_dataset_name", "Desc_Split_dataset", "Name of the dataset to upload the outout to the server. Also used for the next step.")

def pre_processing_callback(pipeline, node, param_override) -> bool:
    print("Cloning step5_desc_split_data id={}".format(node.base_task_id))    
    return True

def post_processing_callback(pipeline, node) -> None:
    print("Completed step5_desc_split_data id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="train_val_splitting",
    parents=["base_desc_generation"],
    base_task_project=project_name,
    base_task_name="step5_desc_split_data",
    parameter_override={
        "General/cap_dataset_id": "${base_desc_generation.parameters.General/output_dataset_id}", 
        "General/cap_dataset_name": "${pipeline.cap_dataset_name}",
        "General/output_dataset_name": pipe.get_parameters()["split_dataset_name"],
        "General/random_state": pipe.get_parameters()["random_state"],
        "General/val_size": pipe.get_parameters()["val_size"]
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
)

""" 
STEP 5: Student Model training
"""
split_dataset_id= '',               
split_dataset_name ='Desc_Split_dataset'            
base_dataset_id = ''
base_dataset_name = 'base_dataset_zip'

# model training settings
pipe.add_parameter("split_dataset_id", "", "(Optional) Overitten if previous task is not skipped. If set, ignore split_dataset_name")
pipe.add_parameter("split_dataset_name", "Desc_Split_dataset", "split data name")
pipe.add_parameter("base_dataset_id", "", "latest of base_dataset_zip id")
pipe.add_parameter("base_dataset_name", "base_dataset_zip", "latest of base_dataset_zip name")

def pre_training_callback(pipeline, node, param_override) -> bool:  
    print("Cloning step6_desc_model_training id={}".format(node.base_task_id))    
    return 
            
def post_training_callback(pipeline, node) -> None:
    print("Completed step6_desc_model_training id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="desc_model_training",
    parents=["train_val_splitting"],
    base_task_project=project_name,
    base_task_name="step6_desc_model_training",
    parameter_override={
        "General/split_dataset_id": "${train_val_splitting.parameters.General/output_dataset_id}",
        "General/split_dataset_name": "${pipeline.split_dataset_name}", 
        "General/base_dataset_id": "${pipeline.base_dataset_id}", 
        "General/base_dataset_name": "${pipeline.base_dataset_name}"},
    pre_execute_callback=pre_training_callback,
    post_execute_callback=post_training_callback
)
"""
Step 6: Model hyperparameter optimisation
"""
# model optimisation settings
pipe.add_parameter("base_train_task_id", "", "base train task")
pipe.add_parameter("time_limit_minutes", 60.0, "Maximum optimization time limit in minutes")
pipe.add_parameter("num_epochs", [2, 3], "list of epochs")
pipe.add_parameter("batch_size", [16, 32], "list of batch size")
pipe.add_parameter("lr", [1e-5, 5e-5, 1e-4], "list of learning rate")
pipe.add_parameter("weight_decay", [1e-3,1e-2], "weight decay values in list")

def pre_hpo_callback(pipeline, node, param_override) -> bool:  
    print("Cloning step7_desc_model_hpo id={}".format(node.base_task_id))    
    print("Cloning Task id={} with parameters: {}".format(
        node.base_task_id, param_override))
    return True
            
def post_hpo_callback(pipeline, node) -> None:
    print("Completed step7_desc_model_hpo id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="desc_model_hpo",
    parents=["desc_model_training"],
    base_task_project=project_name,
    base_task_name="step7_desc_model_hpo",
    parameter_override={
        "General/base_train_task_id": "${desc_model_training.id}",   
        "General/time_limit_minutes": pipe.get_parameters()["time_limit_minutes"],       
        "General/num_epochs": pipe.get_parameters()["num_epochs"],
        "General/batch_size": pipe.get_parameters()["batch_size"],
        "General/lr": pipe.get_parameters()["lr"],
        "General/weight_decay": pipe.get_parameters()["weight_decay"]
    },
    pre_execute_callback=pre_hpo_callback,
    post_execute_callback=post_hpo_callback
)

""" 
STEP 7: Test Data Reference description generation
"""
dataset_id = ""
dataset_name = "Desc_Eval_Dataset"
eval_dataset_id = ''
eval_dataset_name = "eval_dataset_zip"

pipe.add_parameter("dataset_id", "", "latest id of eval data img-label mapping from step 2")
pipe.add_parameter("dataset_name", "Desc_Eval_Dataset", "latest of eval data img-label name from step 2")
pipe.add_parameter("eval_dataset_id", "", "latest of eval_dataset_zip id")
pipe.add_parameter("eval_dataset_name", "eval_dataset_zip", "latest of eval_dataset_zip name")

def pre_processing_callback(pipeline, node, param_override) -> bool:
    print("Cloning step4_desc_evalcaption_generation id={}".format(node.base_task_id))    
    return True
def post_processing_callback(pipeline, node) -> None:
    print("Completed step4_desc_evalcaption_generation id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="eval_desc_generation",
    parents=["EvalData_Mapping", "desc_model_hpo"],
    base_task_project=project_name,
    base_task_name="step4_desc_evalcaption_generation",
    parameter_override={
        "General/dataset_id": "${EvalData_Mapping.parameters.General/output_dataset_id}",
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/eval_dataset_id": "${pipeline.eval_dataset_id}", 
        "General/eval_dataset_name": "${pipeline.eval_dataset_name}"
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
)


"""
STEP 8: Model Evaluation
"""
dataset_id= '',              
dataset_name= 'Desc_Caption_EvalDataset ',     # latest registered dataset
eval_dataset_id= '',
eval_dataset_name= 'eval_dataset_zip',
desc_draft_model_id= '',       # the unpublished model to evaluate 
desc_pub_model_name= 'student_desc_model'
eval_batch_size= 16

pipe.add_parameter("eval_dataset_id", "", "Overitten if previous task is not skipped. If set, ignore eval_dataset_name")
pipe.add_parameter("eval_dataset_name", "eval_dataset_zip", "latest eval image dataset name")
pipe.add_parameter("dataset_id", "", "latest eval caption dataset name")
pipe.add_parameter("dataset_name", "Desc_Caption_EvalDataset", "latest eval caption dataset name")
pipe.add_parameter("desc_draft_model_id", "", "latest trained model in draft state")
pipe.add_parameter("desc_pub_model_name", "student_desc_model", "latest best model in published state")
pipe.add_parameter("eval_batch_size", int(16), "eval batch size")

def pre_eval_callback(pipeline, node, param_override) -> bool:    
    print("Cloning step8_desc_model_evaluation id={}".format(node.base_task_id))      # param validation check
    return True

def post_eval_callback(pipeline, node) -> None:   
    print("Completed step8_desc_model_evaluation id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="desc_model_evaluation",
    parents=["eval_desc_generation", "desc_model_hpo"],
    base_task_project=project_name,
    base_task_name="step8_desc_model_evaluation",
    parameter_override={
        "General/dataset_id": "${eval_desc_generation.parameters.General/output_dataset_id}", 
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/eval_dataset_id": "${pipeline.eval_dataset_id}", 
        "General/eval_dataset_name": "${pipeline.eval_dataset_name}",
        "General/desc_draft_model_id": "${desc_model_hpo.parameters.General/best_model_id}",
        "General/pub_model_name": "${pipeline.desc_pub_model_name}",
        "General/eval_batch_size": "${pipeline.eval_batch_size}"
    },
    pre_execute_callback=pre_eval_callback,
    post_execute_callback=post_eval_callback
)

"""
STEP 9: Model Publishing
"""
def pre_pub_callback(pipeline, node, param_override) -> bool:
    print("Cloning step9_desc_model_publish id={}".format(node.base_task_id))    
    return True

def post_pub_callback(pipeline, node) -> None:
    print("Completed step9_desc_model_publish id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="desc_model_publishing",
    parents=["desc_model_evaluation"],
    base_task_project=project_name,
    base_task_name="step9_desc_model_publish",
    parameter_override={
        "General/desc_draft_model_id": "${desc_model_evaluation.parameters.General/best_model_id}"
    },
    pre_execute_callback=pre_pub_callback,
    post_execute_callback=post_pub_callback
)

remote_execution = project.get("pipeline-remote-execution")
if remote_execution:
    print(f"Executing '{pipeline_name}' pipeline remotely")
    pipe.start(queue = "desc_preparation")
else:
    print(f"Executing '{pipeline_name}' pipeline locally")
    pipe.start_locally(run_pipeline_steps_locally=True)
print("done")