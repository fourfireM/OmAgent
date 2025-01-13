from omagent_core.advanced_components.workflow.ToT.schemas.ToT_structure import ThoughtTree
from omagent_core.utils.registry import registry
from omagent_core.engine.worker.base import BaseWorker

@registry.register_worker()
class ThoughtDecomposition(BaseWorker):
    """
    最基础的任务设置，
    
    尝试模型直接分解任务
    
    """
    
    def _run(self, task: str, query: str):
        # print(self.params)
        if self.stm(self.workflow_instance_id).get('thought_tree', None) is None:
            thought_tree = ThoughtTree()
            thought_tree.add_node(content=query, next_step_input=query, parent_id=None)
            self.stm(self.workflow_instance_id)['task'] = task
            self.stm(self.workflow_instance_id)['thought_tree'] = thought_tree
            self.stm(self.workflow_instance_id)['current_depth'] = 0
            self.stm(self.workflow_instance_id)['current_step'] = 0
            self.stm(self.workflow_instance_id)['current_node_id'] = 0
            self.stm(self.workflow_instance_id)['search_type'] = self.params['search_type']
            self.stm(self.workflow_instance_id)['dfs_best'] = {"id": 0, "score": 0} 
            self.stm(self.workflow_instance_id)['max_depth'] = self.params['max_depth']
            self.stm(self.workflow_instance_id)['max_steps'] = self.params['max_steps']
            self.stm(self.workflow_instance_id)['b'] = self.params['b']
        
        tree_log = {}
        tree_log["0"] = {
            "thought_tree": self.stm(self.workflow_instance_id)['thought_tree'].thought_tree_to_dict(),
        }
        self.stm(self.workflow_instance_id)['tree_log'] = tree_log
        
        # message = ''
        # for key, value in self.params.items():
        #     message += f'{key}: {value}\n'
        # self.callback.info(
        #     agent_id=self.workflow_instance_id,
        #     progress=f"Thought Decomposition",
        #     message='\n'+message,
        # )





        




