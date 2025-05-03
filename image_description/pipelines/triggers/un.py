
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
        "General/dataset_id": "${pipeline.dataset_id}",
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/base_dataset_id": "${pipeline.base_dataset_id}", 
        "General/base_dataset_name": "${pipeline.base_dataset_name}"
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
)

""" 
STEP 4: Test Data Reference description generation
"""
dataset_id = ""
dataset_name = "Desc_Eval_Dataset"
eval_dataset_id = ''
eval_dataset_name = "eval_dataset_zip"

pipe.add_parameter("dataset_id", dataset_id, "latest id of eval data img-label mapping from step 2")
pipe.add_parameter("dataset_name", dataset_name, "latest of eval data img-label name from step 2")
pipe.add_parameter("eval_dataset_id", eval_dataset_id, "latest of eval_dataset_zip id")
pipe.add_parameter("eval_dataset_name", eval_dataset_name, "latest of eval_dataset_zip name")

def pre_processing_callback(pipeline, node, param_override) -> bool:
    print("Cloning step4_desc_evalcaption_generation id={}".format(node.base_task_id))    
    return True
def post_processing_callback(pipeline, node) -> None:
    print("Completed step4_desc_evalcaption_generation id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="eval_desc_generation",
    parents=["EvalData_Mapping"],
    base_task_project=project_name,
    base_task_name="step4_desc_evalcaption_generation",
    parameter_override={
        "General/dataset_id": "${pipeline.dataset_id}",
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/eval_dataset_id": "${pipeline.eval_dataset_id}", 
        "General/eval_dataset_name": "${pipeline.eval_dataset_name}"
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
)
""" 
STEP 5: Splitting Train Dataset
"""
# it will get dataset_id from step 3, if not provided, this will be used
params = {
    'cap_dataset_id': '',
    'cap_dataset_name': 'Desc_Caption_BaseDataset',
    'random_state': 42,
    'val_size': 0.2,
}
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
        "General/cap_dataset_id": "${pipeline.cap_dataset_id}", 
        "General/cap_dataset_name": "${pipeline.cap_dataset_name}",
        "General/output_dataset_name": pipe.get_parameters()["split_dataset_name"],
        "General/random_state": pipe.get_parameters()["random_state"],
        "General/val_size": pipe.get_parameters()["val_size"]
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
)

""" 
STEP 6: Student Model training
"""
""" 
def load_hyp_config(model_variant) -> dict:
    hyp_config_file = f"{model_variant}_hyp_config.yaml"
    hyp_config_path = Path(__file__).parent / hyp_config_file
    print("hyp_config_path=", hyp_config_path.resolve())
    if hyp_config_path.exists():    
        with open(hyp_config_path, "r") as file:
            hyperparameters = yaml.safe_load(file)
    return hyperparameters
"""
split_dataset_id= '',               
split_dataset_name ='Desc_Split_dataset'            
base_dataset_id = ''
base_dataset_name = 'base_dataset_zip'

# model training settings
pipe.add_parameter("split_dataset_id", "", "(Optional) Overitten if previous task is not skipped. If set, ignore split_dataset_name")
pipe.add_parameter("split_dataset_name", split_dataset_name, "split data name")
pipe.add_parameter("base_dataset_id", base_dataset_id, "latest of base_dataset_zip id")
pipe.add_parameter("base_dataset_name", base_dataset_name, "latest of base_dataset_zip name")

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
        "General/split_dataset_id": "${pipeline.split_dataset_id}",   
        "General/split_dataset_name": "${pipeline.split_dataset_name}", 
        "General/base_dataset_id": "${pipeline.base_dataset_id}", 
        "General/base_dataset_name": "${pipeline.base_dataset_name}"},
    pre_execute_callback=pre_training_callback,
    post_execute_callback=post_training_callback
)

"""
STEP 7: Model Evaluation
"""
"""
def load_eval_config(model_variant) -> dict:
    eval_config_file = f"{model_variant}_eval_config.yaml"
    eval_config_path = Path(__file__).parent / eval_config_file
    print("eval_config_path=", eval_config_path.resolve())
    if eval_config_path.exists():    
        with open(eval_config_path, "r") as file:
            eval_confg = yaml.safe_load(file)
    
    return eval_confg
"""
dataset_id= '',              
dataset_name= 'Desc_Caption_EvalDataset ',              # latest registered dataset
eval_dataset_id= '',
eval_dataset_name= 'eval_dataset_zip',
desc_draft_model_id= '',       # the unpublished model to evaluate 
desc_pub_model_name= 'student_desc_model'

pipe.add_parameter("eval_dataset_id", eval_dataset_id, "Overitten if previous task is not skipped. If set, ignore eval_dataset_name")
pipe.add_parameter("eval_dataset_name", eval_dataset_name, "latest eval image dataset name")
pipe.add_parameter("dataset_id", dataset_id, "latest eval caption dataset name")
pipe.add_parameter("dataset_name", dataset_name, "latest eval caption dataset name")
pipe.add_parameter("desc_draft_model_id", desc_draft_model_id, "latest trained model in draft state")
pipe.add_parameter("desc_pub_model_name", desc_pub_model_name, "latest best model in published state")

def pre_eval_callback(pipeline, node, param_override) -> bool:    
    print("Cloning step7_desc_model_evaluation id={}".format(node.base_task_id))      # param validation check
    return True

def post_eval_callback(pipeline, node) -> None:   
    print("Completed step7_desc_model_evaluation id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="desc_model_evaluation",
    parents=["desc_model_training", "eval_desc_generation"],
    base_task_project=project_name,
    base_task_name="step7_desc_model_evaluation",
    parameter_override={
        "General/dataset_id": "${pipeline.dataset_id}", 
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/eval_dataset_id": "${pipeline.eval_dataset_id}", 
        "General/eval_dataset_name": "${pipeline.eval_dataset_name}",
        "General/draft_model_id": "${desc_model_training.parameters.Args/output_model_id}",
        "General/pub_model_name": "${pipeline.desc_pub_model_name}"
    },
    pre_execute_callback=pre_eval_callback,
    post_execute_callback=post_eval_callback
)

"""
STEP 8: Model Publishing
"""
def pre_pub_callback(pipeline, node, param_override) -> bool:
    print("Cloning step8_desc_model_publish id={}".format(node.base_task_id))    
    return True

def post_pub_callback(pipeline, node) -> None:
    print("Completed step8_desc_model_publish id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="desc_model_publishing",
    parents=["desc_model_evaluation"],
    base_task_project=project_name,
    base_task_name="step8_desc_model_publish",
    parameter_override={
        "General/desc_draft_model_id": "${desc_model_evaluation.parameters.Args/best_model_id}"
    },
    pre_execute_callback=pre_pub_callback,
    post_execute_callback=post_pub_callback
)