import numpy as np
import networkx as nx
from tqdm import tqdm
import itertools
from heapq import heappop, heappush
import pdb

class PriorityQueue:
    pq = []                         # list of entries arranged in a heap
    entry_finder = {}               # mapping of tasks to entries
    REMOVED = '<removed-task>'      # placeholder for a removed task
    counter = itertools.count()     # unique sequence count

    def __init__(self):
        pass

    def add_task(self, task, dependencies, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [dependencies, priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def increment_priority(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.remove_task(task)
        priority = entry[1] + 1
        count = next(self.counter)
        entry = [entry[0], priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def decrement_priority(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.remove_task(task)
        priority = entry[1] - 1
        count = next(self.counter)
        entry = [entry[0], priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def decrement(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.remove_task(task)
        deps = entry[0] - 1
        priority = entry[1] - 1
        count = next(self.counter)
        entry = [deps, priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED
        return entry

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            deps, priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                assert deps == 0
                del self.entry_finder[task]
                return task
        return None

def reorder_nodes(graph: nx.DiGraph, window:int = 8) -> list[int]:
    queue = PriorityQueue()
    print('[INFO] Initializing GO algorithm, adding nodes to queue')
    for (u,v) in graph.edges:
        queue.add_task((u,v), graph.in_degree(u))
    print('[INFO] Begining GO Algorithm')

    order = []
    ntasks = len(queue.pq)
    for i in tqdm(range(ntasks)):
        (u,v) = queue.pop_task()
        order.append((u,v))

        for w in graph.successors(v):
            queue.decrement((v,w))
            
        for w in graph.predecessors(v):
            if (w,v) in queue.entry_finder:
                queue.decrement_priority((w,v))
        
        for w in graph.successors(u):
            if (u,w) in queue.entry_finder:
                queue.decrement_priority((u,w))

        if i > window:
            (u_bad, v_bad) = order[-window-1] 
            for w in graph.successors(v_bad):
                if (v_bad, w) in queue.entry_finder:
                    queue.increment_priority((v_bad,w))
                
            for w in graph.predecessors(v_bad):
                if (w,v_bad) in queue.entry_finder:
                    queue.decrement_priority((w,v_bad))
            
            for w in graph.successors(u_bad):
                if (u_bad,w) in queue.entry_finder:
                    queue.decrement_priority((u_bad,w))

    order = [(u, v, graph.edges[u,v][2]) for (u,v) in order]    
    return order
