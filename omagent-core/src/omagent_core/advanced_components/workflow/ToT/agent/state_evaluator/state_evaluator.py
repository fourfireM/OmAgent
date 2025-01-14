from pathlib import Path
from typing import List

import json_repair
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.utils.registry import registry
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import Field
from typing import List

CURRENT_PATH = Path(__file__).parents[0]
    
@registry.register_worker()
class StateEvaluator(BaseWorker, BaseLLMBackend):
    llm: OpenaiGPTLLM
    
    prompts: List[PromptTemplate] = Field([])
    
    value_dict: dict = {
        "sure": 3,
        "likely": 0.5,
        "impossible": -2,
    }
    
    def _run(self, examples: str, *args, **kwargs):
        thought_tree = self.stm(self.workflow_instance_id)['thought_tree']
        current_depth = self.stm(self.workflow_instance_id)['current_depth']
        current_node_id = self.stm(self.workflow_instance_id)['current_node_id']
        
        evaluation_n = self.params['evaluation_n']
        evaluation_type = self.params['evaluation_type']
        
        search_type = self.stm(self.workflow_instance_id)['search_type']

        if not self.prompts:
            self.prompts = [
                PromptTemplate.from_file(
                    CURRENT_PATH.joinpath(f"{evaluation_type}_sys.prompt"), role="system"
                ),
                PromptTemplate.from_file(
                    CURRENT_PATH.joinpath(f"{evaluation_type}_user.prompt"), role="user"
                ),
            ]
        
        if search_type == "bfs":
            current_nodes = thought_tree.get_nodes_at_depth(current_depth)
        elif search_type == "dfs":
            current_nodes = [thought_tree.nodes[current_node_id]]
        else:
            raise ValueError(f"Invalid search type: {search_type}")
        
        #----------------------------------------
        
        # for batch test
        record = self.stm(self.workflow_instance_id)['record']
        prompt_token = record['prompt_token']
        completion_token = record['completion_token']
        evaluation_log = []
        
        if evaluation_type == "value":
            for node in current_nodes:
                evaluation_value = 0
                for i in range(evaluation_n):
                    if node.next_step_input:
                        value_input = node.next_step_input
                    else:
                        value_input = thought_tree.get_current_path_contents(node.id)
                        
                    payload = {
                        "examples": examples,
                        "task": self.stm(self.workflow_instance_id)['task'],
                        "input": value_input,
                    }
                    chat_complete_res = self.infer(input_list=[payload])
                    response = chat_complete_res[0]["choices"][0]["message"].get("content")
                    
                    # for batch test
                    if "usage" in chat_complete_res[0]:
                        prompt_token += chat_complete_res[0]["usage"]["prompt_tokens"]
                        completion_token += chat_complete_res[0]["usage"]["completion_tokens"]
                    
                    # self.callback.info(
                    #     agent_id=self.workflow_instance_id,
                    #     progress=f"State Evaluator-value",
                    #     message=f"input: {value_input}\nresponse: {response}"
                    # )
                    
                    contents = json_repair.loads(response)
                    
                    #-----------------------------------
                    evaluation_log.append(contents)
                    
                    value = contents.get('value', None)
                    evaluation_value += self.value_dict.get(value, 0.0)
                node.value = evaluation_value
                
                
        elif evaluation_type == "vote":
            
            for i in range(evaluation_n):
                choices = ""
                index_to_node_id = {}
                for index, node in enumerate(current_nodes):
                    index_to_node_id[index+1] = node.id
                    if node.next_step_input:
                        choices += f"Choice {index+1}: \n{node.next_step_input}\n"
                    else:
                        next_step_input =  thought_tree.get_current_path_contents(node.id)
                        choices += f"Choice {index+1}: \n{next_step_input}\n"
                    
                payload = {
                    "examples": examples,
                    "task": self.stm(self.workflow_instance_id)['task'],
                    "choices": choices,
                }
                chat_complete_res = self.infer(input_list=[payload])
                response = chat_complete_res[0]["choices"][0]["message"].get("content")
                
                
                # for batch test
                if "usage" in chat_complete_res[0]:
                    prompt_token += chat_complete_res[0]["usage"]["prompt_tokens"]
                    completion_token += chat_complete_res[0]["usage"]["completion_tokens"]
                # self.callback.info(
                #     agent_id=self.workflow_instance_id,
                #     progress=f"State Evaluator-vote",
                #     message=f"choices: {choices}\nresponse: {response}"
                # )
                
                # -----------------------------------------------
                vote_results = json_repair.loads(response)
                
                evaluation_log.append(vote_results)
                choice = vote_results['choice']
                choice_id = index_to_node_id[choice]
                thought_tree.nodes[choice_id].value += 1
        else:
            raise ValueError(f"Invalid evaluation type: {evaluation_type}")

        
        self.stm(self.workflow_instance_id)['thought_tree'] = thought_tree
        
        #=============================================================
        # for batch test
        record['prompt_token'] = prompt_token
        record['completion_token'] = completion_token
        
        self.stm(self.workflow_instance_id)['record'] = record
        
        
        # tree_log = self.stm(self.workflow_instance_id)['tree_log']
        # current_depth_log = tree_log.get(current_depth, {})
        # current_depth_log['evaluation_log'] = evaluation_log
        # tree_log[current_depth] = current_depth_log
        # self.stm(self.workflow_instance_id)['tree_log'] = tree_log
        #=============================================================



