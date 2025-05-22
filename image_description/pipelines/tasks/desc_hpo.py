import sys
import os
from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer, GridSearch
from clearml.automation import DiscreteParameterRange, UniformIntegerParameterRange, UniformParameterRange
import logging
import time
import json
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory
from enigmaai import util
import subprocess
# Install absl-py on the fly so evaluate.load("rouge") can import it
subprocess.check_call([sys.executable, "-m", "pip", "install", "absl-py"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboardX"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocoevalcap"])

# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(project_name=project_name, 
                task_name="step7_desc_model_hpo", 
                task_type=Task.TaskTypes.optimizer,
                reuse_last_task_id=False
)

params = {
    'base_train_task_id': 'e75ef1f7bfb14622a218b1e7f09ae08e', 
    'run_as_service': False,
    'test_queue': 'desc_preparation',  
    'num_epochs': [1, 2], 
    'batch_size': [16],
    'lr': [1e-4],
    'weight_decay': [1e-2]  # Default weight decay
}

params = task.connect(params)
task_params = task.get_parameters()
task.execute_remotely(queue_name=project.get('queue-gpu'))
logger.info(f"model_HPO params={task_params}")

base_task_id = task_params['General/base_train_task_id']

# Exit if not base task
if not base_task_id:
    task.mark_completed(status_message="No base train task ID provided. Nothing to optimisation from.")
    exit(0)

# Create the HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        DiscreteParameterRange('General/num_epochs', values=ast.literal_eval(task_params['General/num_epochs'])),
        DiscreteParameterRange('General/batch_size', values=ast.literal_eval(task_params['General/batch_size'])), 
        DiscreteParameterRange('General/lr', values=ast.literal_eval(task_params['General/lr'])),  
        DiscreteParameterRange('General/weight_decay', values=ast.literal_eval(task_params['General/weight_decay']))],
    objective_metric_title='validation',
    objective_metric_series='cider',
    objective_metric_sign='max',
    optimizer_class=GridSearch,
    max_number_of_concurrent_tasks=5,
    pool_period_min=30.0,
    execution_queue=project.get('queue-gpu'),
    save_top_k_tasks_only=1,
    parameter_override={
        'test_queue': task_params['General/test_queue'],
        'General/test_queue': task_params['General/test_queue'],
        'num_epochs': ast.literal_eval(task_params['General/num_epochs']),
        'General/num_epochs': ast.literal_eval(task_params['General/num_epochs']),
        'batch_size': ast.literal_eval(task_params['General/batch_size']),
        'General/batch_size': ast.literal_eval(task_params['General/batch_size']),
        'lr': ast.literal_eval(task_params['General/lr']),
        'General/lr': ast.literal_eval(task_params['General/lr']),
        'weight_decay': ast.literal_eval(task_params['General/weight_decay']),
        'General/weight_decay': ast.literal_eval(task_params['General/weight_decay'])
    }
    )
hpo_task.set_report_period(30.0)
# Get the top performing experiments
def on_job_complete(job_id, objective_value, objective_iteration, job_parameters, top_performance_job_id):
    best = hpo_task.get_top_experiments(top_k=1)[0]
    logger.info(f"Best so far: {best.id}")
    bp = best.get_parameters()
    mts = best.get_all_reported_scalars().get('validation', {})
    best_cider = mts.get('cider')
    task.upload_artifact('best_parameters', {'parameters': bp, 'best_metrics': best_cider})
    bm = best.models.output[0]
    task.set_parameter("General/best_model_id", bm.id)


# Start the HPO task
logger.info("Starting HPO task...")
remote_execution = True #project.get("pipeline-remote-execution")
hpo_task.start(job_complete_callback=on_job_complete)        
# wait until optimization completed or timed-out
hpo_task.wait()
# make sure we stop all jobs
hpo_task.stop()

"""
def get_top_task_exp(job_id, objective_value, objective_iteration, 
                     job_parameters,top_performance_job_id):
    best_task = hpo_task.get_top_experiments(top_k=1)[0] 
    logger.info(f"Best experiment: {best_task.id}")
    # Get the best parameters and accuracy
    best_params = best_task.get_parameters()
    metrics = best_task.get_all_reported_scalars()
    best_cider = metrics['validation']['cider'] if metrics and 'validation' in metrics and 'cider' in metrics['validation'] else None
    # Save best parameters and accuracy
    best_results = {
        'parameters': best_params,
        'best_metrics': best_cider}
    # Upload as artifact
    task.upload_artifact('best_parameters', best_results)
    print("best results:", best_results)    
    # task output info
    best_model = best_task.models.output[0]
    task.set_parameter("best_model_project", project_name)
    task.set_parameter("best_model_task_id", best_model.name)
    task.set_parameter("best_model_id", best_model.id)
if remote_execution:
    if hpo_task.start(job_complete_callback=get_top_task_exp):
        print(f"Executing HPO remotely")
    else:
        print("HPO failed to start remotely")
else:
    print(f"Executing HPO locally")
    if hpo_task.start_locally(job_complete_callback=get_top_task_exp):
        print(f"Executing HPO locally")
    else:
        print("HPO failed to start locally")"""