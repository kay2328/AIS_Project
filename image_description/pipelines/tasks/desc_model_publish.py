import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from clearml import Task, Model
from enigmaai.config import Project, ConfigFactory

"""
Publish a specific draft model from the server. The model can be in Draft for Published state.
If it is already in Published state, the model will not be published again. Refer to Model.publish() API.
"""

# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')
task = Task.init(project_name=project_name, 
                task_name="step8_desc_model_publish", 
                task_type=Task.TaskTypes.qc)

params = {
    'desc_draft_model_id': '6d7b1be6082543efaade2fb24df63ae3' #'96f429eb382f44b1a08a78e168c7bf3b',      # specific id of the model
}

task.connect(params)
task.execute_remotely(queue_name="desc_preparation")
task_params = task.get_parameters()
print("model_publish params=", task_params)

draft_model_id = params['desc_draft_model_id']
    
# no model provided for publishing
if not draft_model_id:
    task.mark_completed(status_message="No draft model provided for publishing.")
    exit(0)
# fetch the specific draft model   
draft_model = Model(model_id=draft_model_id)    
print(f"Found draft model name:{draft_model.name} id:{draft_model.id}")

# publish the model
if not draft_model.published:
    print(f"Publishing draft model name:{draft_model.name} id:{draft_model.id}")
    draft_model.publish()
    print(f"Draft model successfully published. New published model name:{draft_model.name} id:{draft_model.id}")
    print("Done")
else: 
    print(f"Model already published with name:{draft_model.name} id:{draft_model.id}")

# show output    
print("pub_model_project:", project_name)
print("pub_model_id:", draft_model.id)
print("pub_model_name:", draft_model.name)

# task output info
task.set_parameter("pub_model_project", project_name)
task.set_parameter("pub_model_id", draft_model.id)
task.set_parameter("pub_model_name", draft_model.name)