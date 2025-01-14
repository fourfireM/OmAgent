from pathlib import Path
from typing import List
import json
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.utils.registry import registry
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import Field
from typing import List
import json_repair

CURRENT_PATH = Path(__file__).parents[0]

'''
加一个判断任务是否结束

'''
@registry.register_worker()
class SearchAlgorithm(BaseWorker, BaseLLMBackend):
    
    llm: OpenaiGPTLLM
    
    def _run(self, *args, **kwargs):
        thought_tree = self.stm(self.workflow_instance_id).get('thought_tree', None)
        current_depth = self.stm(self.workflow_instance_id).get('current_depth', None)
        current_step = self.stm(self.workflow_instance_id).get('current_step', None)

        max_depth = self.stm(self.workflow_instance_id).get('max_depth', None)
        max_steps = self.stm(self.workflow_instance_id).get('max_steps', None)
        dfs_best = self.stm(self.workflow_instance_id).get('dfs_best', None)
        
        current_node_id = self.stm(self.workflow_instance_id).get('current_node_id', None)
        

        search_type = self.stm(self.workflow_instance_id).get('search_type', None)
        
        # self.callback.info(
        #     agent_id=self.workflow_instance_id,
        #     progress=f"Search Algorithm",
        #     message=f"thought_tree: {thought_tree}\n"
        # )
        
        # for batch test
        record = self.stm(self.workflow_instance_id)['record']
        prompt_token = record['prompt_token']
        completion_token = record['completion_token']
        
        
        if search_type == "bfs":
            b = self.stm(self.workflow_instance_id).get('b', None)
            current_nodes_ids = thought_tree.get_nodes_at_depth(current_depth, return_ids=True)
            top_b_nodes_ids = thought_tree.get_top_n_score_nodes(depth=current_depth, sort_region='depth', n=b, return_ids=True)
            for node_id in current_nodes_ids:
                if node_id not in top_b_nodes_ids:
                    thought_tree.prune(node_id)
            best_node_id = top_b_nodes_ids[0]
        elif search_type == "dfs":
            current_node_score = thought_tree.nodes[current_node_id].value
            current_path_score = thought_tree.get_current_path_score(current_node_id)
            if current_node_score < 0:
                prune_id = current_node_id
                parent_id = thought_tree.get_parent(current_node_id, return_ids=True)
                thought_tree.prune(prune_id)
                current_depth -= 1
                while thought_tree.get_childrens(parent_id) == []:
                    prune_id = parent_id
                    parent_id = thought_tree.get_parent(parent_id, return_ids=True)
                    thought_tree.prune(prune_id)
                    current_depth -= 1
                current_node_id = parent_id
            else:
                if dfs_best['score'] < current_path_score:
                    dfs_best['id'], dfs_best['score'] = current_node_id, current_path_score
                best_node_id = dfs_best['id']

        else:
            raise ValueError(f"Search type {search_type} is not supported")
        
        # self.callback.info(
        #     agent_id=self.workflow_instance_id,
        #     progress=f"Search Algorithm",
        #     message=f"thought_tree_after_prune: {thought_tree}\n"
        # )
        
        self.stm(self.workflow_instance_id)['thought_tree'] = thought_tree
        self.stm(self.workflow_instance_id)['current_node_id'] = current_node_id
        self.stm(self.workflow_instance_id)['current_depth'] = current_depth
        
        # tree_log = self.stm(self.workflow_instance_id)['tree_log']
        # current_depth_log = tree_log.get(current_depth, {})
        # current_depth_log['pruned_tree'] = thought_tree.thought_tree_to_dict()
        # tree_log[current_depth] = current_depth_log
        # self.stm(self.workflow_instance_id)['tree_log'] = tree_log
        
        # self.callback.info(
        #     agent_id=self.workflow_instance_id,
        #     progress=f"Search Algorithm",
        #     message=f"Tree log saved to tree_log.json"
        # )

        
        current_best_contents = thought_tree.get_current_path_contents(node_id=best_node_id)
        # self.callback.info(
        #     agent_id=self.workflow_instance_id,
        #     progress=f"Search Algorithm-completion",
        #     message=f'record: {record}\n '
        # )
        
        if self.use_llm_completion:
            payload = {
                "Input": current_best_contents
            }
        
            chat_complete_res = self.infer(input_list=[payload])
            response = chat_complete_res[0]["choices"][0]["message"].get("content")
            
            response = json_repair.loads(response)
            
            # ========================================================================
            
            # for batch test
            if "usage" in chat_complete_res[0]:
                prompt_token += chat_complete_res[0]["usage"]["prompt_tokens"]
                completion_token += chat_complete_res[0]["usage"]["completion_tokens"]
            
            # for batch test
            record['prompt_token'] = prompt_token
            record['completion_token'] = completion_token
            
            self.stm(self.workflow_instance_id)['record'] = record
            
            
            # tree_log[current_depth]['completion'] = response
            
            # self.stm(self.workflow_instance_id)['tree_log'] = tree_log
            # Convert the tree log to a JSON string

                
            # ========================================================================
            
            # self.callback.info(
            #     agent_id=self.workflow_instance_id,
            #     progress=f"Search Algorithm-completion",
            #     message=f'response: {response}\n '
            # )
        
            if response.get('completion', None) == "yes":
                # for batch test
                record = self.stm(self.workflow_instance_id)['record']
                record['last_output'] = current_best_contents
                
                # tree_log = self.stm(self.workflow_instance_id)['tree_log']
                # tree_log['record'] = record
                # qid = record['qid']
                
                # tree_log_json = json.dumps(tree_log, indent=4)

                # Save the JSON string to a file
                # file_name = f"/data9/myb/OmAgent-myb/OmAgent/omagent-test-data/GSM8K/tree_log/{qid}.json"
                # with open(file_name, "w") as file:
                #     file.write(tree_log_json)
                
                # self.callback.info(
                #     agent_id=self.workflow_instance_id,
                #     progress=f"Search Algorithm-completion",
                #     message=f'record: {record}\n '
                # )
                return {"finish": True, "result": record}
        
        if current_depth >= max_depth or current_step >= max_steps:
            record = self.stm(self.workflow_instance_id)['record']
            record['last_output'] = current_best_contents   
            return {"finish": True, "result": record}
        else:
            return {"finish": False}
                
        
            
            
    