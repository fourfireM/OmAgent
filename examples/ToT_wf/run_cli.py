# Import required modules and components
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.advanced_components.workflow.ToT.workflow import ToTWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.cli.client import DefaultClient
from omagent_core.engine.workflow.task.set_variable_task import SetVariableTask
from omagent_core.utils.logger import logging
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask

# from agent.input_interface.input_interface import InputInterface

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

container.register_stm("RedisSTM")
# Load container configuration from YAML file
container.from_config(CURRENT_PATH.joinpath('container.yaml'))



# Initialize simple VQA workflow
workflow = ConductorWorkflow(name='ToT_Workflow')

tot_workflow = ToTWorkflow()
tot_workflow.set_tot(
    task = "Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with 'The answer is'. When the original question is answerable, please start the subquestion with 'Now we can answer the question: '. Remember each time only generate one next sub-question and answer.",
    thought_generator_examples_file_path=CURRENT_PATH.joinpath('prompts/thought_generator/math.examples'),
    state_evaluator_examples_file_path=CURRENT_PATH.joinpath('prompts/state_evaluator/math_vote.examples')
)

tot_workflow.set_input(
    query = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
    id = "test_1"
)



# output_interface = simple_task(task_def_name='OutputInterface', task_reference_name='output_interface')

# Configure workflow execution flow: Input -> TOT_LOOP -> Output
# workflow >> thought_decomposition >> thought_generator >> state_evaluator >> search_algorithm

workflow >> tot_workflow



# Register workflow
workflow.register(True)

# Initialize and start CLI client with workflow configuration
config_path = CURRENT_PATH.joinpath('configs')
cli_client = DefaultClient(interactor=workflow, config_path=config_path, workers=[])
cli_client.start_interactor()
