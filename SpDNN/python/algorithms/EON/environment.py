import networkx as nx

class SparseNetowrkEnv():
    def __init__(self, G: nx.DiGraph, max_frontier_len: 1024, window: int = 8):
        self.G = G
        self.window = window

        self.order = []
        self.cur_graph = self.G.copy()
        self._cache = [-1 for _ in range(window)]
        self._frontier = [e for e in G.edges if G.in_degree(e[0]) == 0]
        self.max_frontier_len = max_frontier_len

    def _get_obs(self):
        return {
            "cache": self._cache.copy(), 
            "frontier": self._frontier.copy() + [(-1, -1)] * (self.max_frontier_len - len(self._frontier))
        }

    def reset(self):
        self.order = []
        self.cur_graph = self.G.copy()
        self._cache = [-1 for _ in range(self.window)]
        self._frontier = [e for e in self.G.edges if self.G.in_degree(e[0]) == 0]

        obs = self._get_obs()
        return obs

    def step(self, action):
        (u,v) = action
        
        reward = 0
        terminated = False
        if (u,v) in self._frontier:
            self.cur_graph.remove_edge(u,v)
            self._frontier.remove((u,v))
            self.order.append((u,v))
            if self.cur_graph.in_degree(v) == 0:
                for w in self.cur_graph.successors(v):
                    self._frontier.append((v,w))

            terminated = nx.is_empty(self.cur_graph)

            misses = 0
            if u in self._cache:
                self._cache.remove(u)
            else:
                misses += 1

            if v in self._cache:
                self._cache.remove(v)
            else:
                misses += 1

            self._cache.extend([u,v])
            self._cache = self._cache[-self.window:]
            reward = -misses
        else:
            # Don't update state, but return large negative reward
            print((u,v), self.cur_graph.number_of_edges(), self._frontier)
            print(list(self.cur_graph.edges))
            raise RuntimeError

            reward = -10

        obs = self._get_obs()
        return obs, reward, terminated