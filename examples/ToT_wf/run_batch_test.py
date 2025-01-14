from typing import Union
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from pathlib import Path, PurePath
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.programmatic.client import ProgrammaticClient
from omagent_core.utils.logger import logging
from omagent_core.advanced_components.workflow.ToT.workflow import ToTWorkflow
import json
import jsonlines
import argparse


def load_jsonl(jsonl_path: str):
    res = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            res.append(item)
    return res

def write_jsonl(data: list, save_path: str, save_name: str):
    with jsonlines.open(PurePath(save_path, f"{save_name}.jsonl"), 'w') as f:
        for _data in data:
            f.write(_data)
            
def write_json(data: Union[list, dict], save_path: str, save_name: str):
    with open(PurePath(save_path, f"{save_name}.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
    return str(PurePath(save_path, f"{save_name}.json"))


def run_workflow(input_datas: list, task: str, thought_generator_examples_file_path: str, state_evaluator_examples_file_path: str):
    """Run the ReAct Pro workflow with given arguments"""
    logging.init_logger("omagent", "omagent", level="INFO")

    # Set current working directory path
    CURRENT_PATH = Path(__file__).parents[0]
    ROOT_PATH = CURRENT_PATH.parents[1]  

    # Import registered modules
    registry.import_module(CURRENT_PATH.joinpath('agent'))

    # Load container configuration from YAML file
    container.register_stm("RedisSTM")
    container.from_config(CURRENT_PATH.joinpath('container.yaml'))

    # Initialize workflow
    workflow = ConductorWorkflow(name='tot_workflow_batch_test')

    # Configure React Pro workflow
    tot_workflow = ToTWorkflow()
    
    
    
    tot_workflow.set_tot(
        task=task,
        thought_generator_examples_file_path=thought_generator_examples_file_path,
        state_evaluator_examples_file_path=state_evaluator_examples_file_path
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
        workers=[]  
    )

    # Prepare input data
    workflow_input_list = [
        {"query": item['question'], "qid": item['id']} for idx, item in enumerate(input_datas)
    ]

    # print(f"Processing {len(workflow_input_list)} queries in this split...")

    # Process data
    res = programmatic_client.start_batch_processor(
        workflow_input_list=workflow_input_list,
    )

    programmatic_client.stop_processor()

    return res

if __name__ == "__main__":
    CURRENT_PATH = Path(__file__).parents[0]
    ROOT_PATH = CURRENT_PATH.parents[1]  
    # print(ROOT_PATH)
    # print(CURRENT_PATH)
    
    unanswerable_ids = load_jsonl('/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K/unanswerable_ids.jsonl')
    test_data_path = '/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K/gsm8k_test.jsonl'
    original_datas = load_jsonl(test_data_path)
    
    input_datas = []
    gt = {}
    for idx, data in enumerate(original_datas):
        if data['id'] in unanswerable_ids:
            data['question'] = f"Question: {data['question']}"
            gt[data['id']] = data['answer']
            input_datas.append(data)
        
    task = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the subquestion with "Now we can answer the question: ". 
Remember each time only generate one next sub-question and answer. 
If the question is simple enough to answer directly, you can answer it in the first subquestion and start with "Now we can answer the question: ".
    """

    thought_generator_examples_file_path = CURRENT_PATH.joinpath('prompts/thought_generator/gsm8k.examples')
    state_evaluator_examples_file_path = CURRENT_PATH.joinpath('prompts/state_evaluator/gsm8k_vote.examples')
    
    for idx in range(0, len(input_datas), 10):  # 每次处理10条数据
        batch_input_datas = input_datas[idx:idx + 10]  # 获取当前批次的数据
        # print(f"Processing batch {idx // 10 + 1} / {len(input_datas) // 10 + 1}...")
        res = run_workflow(batch_input_datas, task, thought_generator_examples_file_path, state_evaluator_examples_file_path)
        
        # 保存结果
        result_file_name = f"batch_{idx // 10 + 1}_result.json"
        write_json(res, "/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K/result", result_file_name)

    # print(res)
    
    
    # final_res = {
    #     "dataset": "GSM8K",
    #     "model_id": "gpt-3.5-turbo",
    #     "alg": "ToT",
    #     "search_type": "bfs",
    #     "temperature": 0.0,
    #     "max_tokens": 2048,
    #     "model_result": []
    # }
    
    # for r in res:
    #     if r is None:
    #         continue
    #     item = r['result']
    #     final_res['model_result'].append({
    #         "id": item['qid'],
    #         "question": item['question'],
    #         "last_output": item['last_output'],
    #         "ground_truth": gt[item['qid']],
    #         "prompt_tokens": item['prompt_token'],
    #         "completion_tokens": item['completion_token']
    #     })
            
    # result_file_name = f"{final_res['dataset']}_{final_res['model_id']}_{final_res['alg']}_{final_res['search_type']}_{final_res['temperature']}_result"

    # write_json(final_res, "/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K/result", result_file_name)
    
