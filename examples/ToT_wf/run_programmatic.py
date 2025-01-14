import jsonlines
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from pathlib import Path, PurePath
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.programmatic.client import ProgrammaticClient
from omagent_core.utils.logger import logging
from omagent_core.advanced_components.workflow.ToT.workflow import ToTWorkflow

logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(CURRENT_PATH.joinpath('agent'))

# Load container configuration from YAML file
container.register_stm("RedisSTM")
container.from_config(CURRENT_PATH.joinpath('container.yaml'))

# Initialize workflow
workflow = ConductorWorkflow(name='tot_example')

# Configure React Pro workflow
tot_workflow = ToTWorkflow()
tot_workflow.set_tot(
    task = "Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with 'The answer is'. When the original question is answerable, please start the subquestion with 'Now we can answer the question: '. Remember each time only generate one next sub-question and answer.",
    thought_generator_examples_file_path=CURRENT_PATH.joinpath('prompts/thought_generator/math.examples'),
    state_evaluator_examples_file_path=CURRENT_PATH.joinpath('prompts/state_evaluator/math_vote.examples')
)


tot_workflow.set_input(
    query=workflow.input('query'),
    qid=workflow.input('qid')
)

# Configure workflow execution flow
workflow >> tot_workflow 

# Register workflow
workflow.register(overwrite=True)

# Initialize programmatic client
config_path = CURRENT_PATH.joinpath('configs')
programmatic_client = ProgrammaticClient(
    processor=workflow,
    config_path=config_path,
    workers=[]  # No additional workers needed for React Pro workflow
)


def load_jsonl(jsonl_path: str):
    res = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            res.append(item)
    return res

input_datas = load_jsonl(CURRENT_PATH.joinpath('/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K/8-examples-q.jsonl'))
# Prepare input data
workflow_input_list = [
    {"query": item['query'], "qid": item['id']} for item in input_datas
]

# print(f"\nProcessing query: {workflow_input_list[0]['query']}")
# print(f"Query ID: {workflow_input_list[0]['qid']}\n")

res = programmatic_client.start_batch_processor(
    workflow_input_list=workflow_input_list
)

programmatic_client.stop_processor()

# Print results
print(res)
def write_jsonl(data: list, save_path: str, save_name: str):
    with jsonlines.open(PurePath(save_path, f"{save_name}.jsonl"), 'w') as f:
        for _data in data:
            f.write(_data)
            
            
write_jsonl(res, '/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K', '8-tot_res')