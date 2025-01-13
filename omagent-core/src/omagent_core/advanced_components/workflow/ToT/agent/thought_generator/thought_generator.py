from pathlib import Path
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
class ThoughtGenerator(BaseWorker, BaseLLMBackend):
    
    llm: OpenaiGPTLLM

    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("sys_prompt.prompt"), role="system"
            ),
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("user_prompt.prompt"), role="user"
            ),
        ]
    )

    def _run(self, examples: str, *args, **kwargs):

        thought_tree = self.stm(self.workflow_instance_id)['thought_tree']
        current_depth = self.stm(self.workflow_instance_id)['current_depth']
        current_node_id = self.stm(self.workflow_instance_id)['current_node_id']
        current_step = self.stm(self.workflow_instance_id)['current_step']
        search_type = self.stm(self.workflow_instance_id)['search_type']
        generation_n = self.params['generation_n']
        
        # generation_log = []

        do_generate = True
        if search_type == "bfs":
            current_nodes = thought_tree.get_nodes_at_depth(current_depth)
        elif search_type == "dfs":
            current_node_children_ids = thought_tree.get_childrens(current_node_id, return_ids=True)
            if current_node_children_ids:
                current_node_id = current_node_children_ids[0]
                do_generate = False
            else:
                current_nodes = [thought_tree.nodes[current_node_id]]
        else:
            raise ValueError(f"Invalid search type: {search_type}")
        
        
        if do_generate:
            for node in current_nodes:
                for _ in range(generation_n):
                    if node.next_step_input is None:
                        next_step_input = thought_tree.get_current_path_contents(node.id)
                    else:
                        next_step_input = node.next_step_input
                        
                    payload = {
                        "examples": examples,
                        "task": self.stm(self.workflow_instance_id)['task'],
                        "input": next_step_input,

                    }
                    chat_complete_res = self.infer(input_list=[payload])
                    response = chat_complete_res[0]["choices"][0]["message"].get("content")
                    response = json_repair.loads(response)

                    # self.callback.info(
                    #     agent_id=self.workflow_instance_id,
                    #     progress=f"Thought Generator",
                    #     message=f'response: {response}'
                    # )
                    
                    # generation_log.append(response)
                    
                    if response.get('llm_response'):
                        for index, llm_response in enumerate(response['llm_response']):
                            next_step_input = response.get('next_step_input', None)
                            if next_step_input and index < len(next_step_input) and isinstance(next_step_input, list):
                                next_step_input = next_step_input[index]
                            
                            thought_tree.add_node(content=llm_response, next_step_input=next_step_input, parent_id=node.id)
        
        if search_type == "dfs":
            self.stm(self.workflow_instance_id)['current_node_id'] = thought_tree.nodes[current_node_id].children[0]
        # before_thought_tree = self.stm(self.workflow_instance_id)['thought_tree']
        # before_current_node_id = self.stm(self.workflow_instance_id)['current_node_id']
        # before_current_step = self.stm(self.workflow_instance_id)['current_step']
        
        self.stm(self.workflow_instance_id)['thought_tree'] = thought_tree
        self.stm(self.workflow_instance_id)['current_depth'] = current_depth + 1
        self.stm(self.workflow_instance_id)['current_node_id'] = current_node_id
        self.stm(self.workflow_instance_id)['current_step'] = current_step + 1
        
        # #=============================================================
        # tree_log = self.stm(self.workflow_instance_id)['tree_log']
        # current_depth_log = tree_log.get(current_depth+1, {})
        # current_depth_log['generation_log'] = generation_log
        # current_depth_log['generation_thought_tree'] = thought_tree.thought_tree_to_dict()
        # tree_log[current_depth+1] = current_depth_log
        # self.stm(self.workflow_instance_id)['tree_log'] = tree_log
        # #=============================================================
        
        
        # print('--show--stm--'*20)
        # print(f'before thought_tree: {before_thought_tree.nodes}')
        # print(f'before current_depth: {current_depth}')
        # print(f'before current_node_id: {before_current_node_id}')
        # print(f'before current_step: {current_step}')
        # print('--------------'*20)
        # print(f'after thought_tree: {self.stm(self.workflow_instance_id)["thought_tree"].nodes}')
        # print(f'after current_depth: {self.stm(self.workflow_instance_id)["current_depth"]}')
        # print(f'after current_node_id: {self.stm(self.workflow_instance_id)["current_node_id"]}')
        # print(f'after current_step: {self.stm(self.workflow_instance_id)["current_step"]}')
        # print('--show--stm--'*20)
        
        

