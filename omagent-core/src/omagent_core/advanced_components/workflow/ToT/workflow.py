from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task

from omagent_core.advanced_components.workflow.ToT.agent.thought_decomposition.thought_decomposition import ThoughtDecomposition
from omagent_core.advanced_components.workflow.ToT.agent.thought_generator.thought_generator import ThoughtGenerator
from omagent_core.advanced_components.workflow.ToT.agent.state_evaluator.state_evaluator import StateEvaluator
from omagent_core.advanced_components.workflow.ToT.agent.search_algorithm.search_algorithm import SearchAlgorithm
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask

class ToTWorkflow(ConductorWorkflow):
    def __init__(self):
        super().__init__(name='tot_workflow')
        
    def set_tot(self, task: str, thought_generator_examples_file_path: str=None, state_evaluator_examples_file_path: str=None):
        self.task = task
        if thought_generator_examples_file_path:
            self.set_generator_examples(thought_generator_examples_file_path)
        else:
            self.thought_generator_examples = None
        if state_evaluator_examples_file_path:
            self.set_evaluator_examples(state_evaluator_examples_file_path)
        else:
            self.state_evaluator_examples = None

        
    def set_input(self, query: str, id: str):
        self.query = query
        self.id = id
        self._configure_tasks()
        self._configure_workflow()
        
    def set_generator_examples(self, examples_file_path: str):
        with open(examples_file_path, 'r') as file:
            self.thought_generator_examples = file.read()
    def set_evaluator_examples(self, examples_file_path: str):
        with open(examples_file_path, 'r') as file:
            self.state_evaluator_examples = file.read()

    def _configure_tasks(self):
        self.thought_decomposition_task = simple_task(
            task_def_name=ThoughtDecomposition,
            task_reference_name="thought_decomposition",
            inputs={
                "query": self.query,
                "task": self.task,
            }
        )
        self.thought_generator_task = simple_task(
            task_def_name=ThoughtGenerator, task_reference_name="thought_generator",
            inputs={
                "examples": self.thought_generator_examples,
            }
        )
        self.state_evaluator_task = simple_task(
            task_def_name=StateEvaluator, task_reference_name="state_evaluator",
            inputs={
                "examples": self.state_evaluator_examples,
            }
        )
        self.search_algorithm_task = simple_task(
            task_def_name=SearchAlgorithm, task_reference_name="search_algorithm"
        )
        
        self.tot_loop_task = DoWhileTask(
            task_ref_name='tot_loop', tasks=[self.thought_generator_task, self.state_evaluator_task, self.search_algorithm_task], 
            termination_condition='if ($.search_algorithm["finish"] == true){false;} else {true;} ')

    def _configure_workflow(self):
        
        # self >> self.thought_decomposition_task
        # self >> self.thought_decomposition_task  >> self.thought_generator_task 
        # self >> self.thought_decomposition_task >> self.state_evaluator_task
        # self >> self.thought_decomposition_task  >> self.thought_generator_task >> self.state_evaluator_task 
        # self >> self.thought_decomposition_task  >> self.thought_generator_task >> self.state_evaluator_task >> self.search_algorithm_task
        self >> self.thought_decomposition_task >> self.tot_loop_task
        self.result = self.search_algorithm_task.output("result")
        
        
        