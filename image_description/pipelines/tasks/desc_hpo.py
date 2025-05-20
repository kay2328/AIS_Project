import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
import logging
import time
import json
import yaml
from enigmaai.config import Project, ConfigFactory
from enigmaai import util

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(project_name=project_name, 
                task_name="Model HPO", 
                task_type=Task.TaskTypes.optimizer,
                reuse_last_task_id=False
)

params = {
    'base_task_id': '', 
    'hpo_min_batch': 2,             # minimum batch size in HPO range
    'hpo_max_batch': 6,             # maximum batch size in HPO range
    'hpo_min_weight_decay': 1e-5,   # minimum weight decay
    'hpo_max_weight_decay': 1e-5,   # maximum weight decay
    'total_max_jobs': 3,            # Total maximum job for the optimization process
    'max_job_iter': 5,              # Number of iteration per job ‘iterations’ for the specified objective
}

task.connect(params)
task_params = task.get_parameters()
task.execute_remotely(queue_name=project.get('queue-gpu'))
logger.info(f"model_HPO params={task_params}")

base_task_id = task_params['General/base_task_id']

# Exit if not base task
if not base_task_id:
    task.mark_completed(status_message="No base task ID provided. Nothing to optimisation from.")
    exit(0)

# Create the HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformIntegerParameterRange('General/batch', 
                                     min_value=int(task_params['General/hpo_min_batch']), 
                                     max_value=int(task_params['General/hpo_max_batch'])), 
        UniformParameterRange('General/weight_decay', 
                              min_value=float(task_params['General/hpo_min_weight_decay']), 
                              max_value=float(task_params['General/hpo_max_weight_decay']))  
    ],
    objective_metric_title='validation',
    objective_metric_series='recall',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=10,
    total_max_jobs=int(task_params['General/total_max_jobs']),
    min_iteration_per_job=1,
    max_iteration_per_job=int(task_params['General/max_job_iter']),
    pool_period_min=10, 
    execution_queue=project.get('queue-gpu'),
    save_top_k_tasks_only=1
)

# Get the top performing experiments
def get_top_task_exp(job_id, objective_value, objective_iteration, 
                     job_parameters,top_performance_job_id):
    
    best_task = hpo_task.get_top_experiments(top_k=1)[0] 
    logger.info(f"Best experiment: {best_task.id}")
    
    # Get the best parameters and accuracy
    best_params = best_task.get_parameters()
    metrics = best_task.get_all_reported_scalars()
    best_recall = metrics['validation']['recall'] if metrics and 'validation' in metrics and 'recall' in metrics['validation'] else None
    
    # Save best parameters and accuracy
    best_results = {
        'parameters': best_params,
        'best_metrics': best_recall
    }
    
    # Upload as artifact
    task.upload_artifact('best_parameters', best_results)
    print("best results:", best_results)
    
    # task output info
    best_model = best_task.models.output[0]
    task.set_parameter("best_model_project", project_name)
    task.set_parameter("best_model_task_id", best_model.name)
    task.set_parameter("best_model_id", best_model.id)
    task.set_parameter("best_model_name", best_model.name)
    task.set_parameter("best_model_variant", best_model.name)
    
# This will automatically create and print the optimizer new task id
# for later use. if a Task was already created, it will use it.
# hpo_task.set_time_limit(in_minutes=720.)

# Start the HPO task
logger.info("Starting HPO task...")
remote_execution = project.get("pipeline-remote-execution")

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
        print("HPO failed to start locally")
        
# wait until optimization completed or timed-out
hpo_task.wait()
# make sure we stop all jobs
hpo_task.stop()