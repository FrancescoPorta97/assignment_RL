import random
import numpy as np

class Environment:

    def __init__(self, resource_capacity: int , num_resources: int):

        self.num_resources = num_resources  
        self.resource_capacities = [resource_capacity]* num_resources
        self.cost_task_resource = self._cost_definition()
        self.tasks_to_mask = []
        self.resource_to_fill = self._init_resource()
        self.filled_resources = []
        self.assignment_solution = self._init_assignment_solution()

    def get_num_tasks(self):

        return sum(self.resource_capacities)
    
    def _cost_definition(self):
        
        cost_task_resources =np.random.random(size=(self.num_resources, self.get_num_tasks() ))
        return cost_task_resources
    
    def _init_resource(self):

        ret = np.zeros(shape=(self.num_resources))
        ret[0] = 1
        return ret
    
    def _init_assignment_solution(self):

        ret = {}
        for idx in range(self.num_resources):
            ret[str(idx)]= []
        
        return ret
    
    def increase_capacity(self, filled_resources):

        remaining_resources = list(set( range(len(self.resource_capacities))) - set(filled_resources))
        idx_to_increase = random.choice(remaining_resources)

        new_capacities = []
        for idx, capacity in enumerate(self.resource_capacities):

            if idx == idx_to_increase:
                new_capacities.append(capacity +1)
            else:
                new_capacities.append(capacity)

        self.cost_task_resource = np.hstack([self.cost_task_resource, np.random.random(size=(self.num_resources,1))])
        self.resource_capacities = new_capacities
    
    def reduce_capacity(self, filled_resources):
        
        remaining_resources =  list(set( range(len(self.resource_capacities))) - set(filled_resources))
        idx_to_reduce = random.choice(remaining_resources)

        new_capacities = []
        for idx, capacity in enumerate(self.resource_capacities):

            if idx == idx_to_reduce:
                new_capacities.append(capacity -1)
            else:
                new_capacities.append(capacity)

        self.resource_capacities = new_capacities
    
    def get_eligible_tasks(self):

        total_tasks = [task for task in range(self.get_num_tasks())]
        eligible_tasks = list(set(total_tasks)-set(self.tasks_to_mask))
        return eligible_tasks
    
    def get_not_available_tasks(self):

        return self.tasks_to_mask

    def is_resource_available(self):

        resource_idx = self.get_current_resource_idx()
        assigned_tasks = self.assignment_solution[str(resource_idx)]
        capacity = self.resource_capacities[resource_idx]
        
        if len(assigned_tasks) == capacity:
            ret = False
        else:
            ret =True
        
        return ret
    
    def take_action(self, action_idx) -> int:

        self.tasks_to_mask.append(action_idx)
        resource_idx = self.get_current_resource_idx()
        self.assignment_solution[str(resource_idx)].append(action_idx)
        reward = self.cost_task_resource[self.get_current_resource_idx(), action_idx]
        return -reward

    def  get_current_resource_idx(self):

        return int(np.where(self.resource_to_fill == 1)[0][0])

    def update_resource(self):

        current_idx = self.get_current_resource_idx()
        self.filled_resources.append(current_idx)
        next_idx = current_idx +1
        ret = np.zeros(shape=(self.num_resources))
        ret[next_idx] = 1
        self.resource_to_fill = ret



    

    
    


    






    

