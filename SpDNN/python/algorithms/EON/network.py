import networkx as nx
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import torch
from torch import nn

class EdgeOrderingNetwork(nn.Module):
    def __init__(self, graph, embedding_dim=8, hidden_dim=64):
        super(EdgeOrderingNetwork, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(self.node_emebddings(graph, embedding_dim))

        self.cache_layer = nn.Linear(embedding_dim, hidden_dim)
        self.actions_layer = nn.Linear(embedding_dim, hidden_dim)

        self.hidden_layer1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)

    def node_emebddings(self, graph, k):
        L = nx.directed_laplacian_matrix(graph)
        (_, e) = sp.sparse.linalg.eigs(L,k=k)
        padding = torch.zeros((1, k))
        embeddings = torch.cat((padding, torch.Tensor(e)))
        return embeddings

    def __call__(self, frontier, cache):
        while frontier.dim() < 3:
            frontier = frontier.unsqueeze(0)
            cache = cache.unsqueeze(0)

        cache_embds = self.embedding(cache + 1)
        cache_embds = torch.sum(self.cache_layer(cache_embds), dim=-2)
        norm = torch.sum((cache >= 0), dim=-1).unsqueeze(-1) + 1
        cache_embds = cache_embds / norm

        frontier_embds = self.embedding(frontier + 1)
        mask = (frontier[:,:,0] >= 0)
        frontier_embds = self.actions_layer(frontier_embds)
        frontier_embds = torch.reshape(frontier_embds, (frontier_embds.shape[0], frontier_embds.shape[1], -1))
        frontier_embds = self.hidden_layer1(frontier_embds)

        logits = torch.bmm(frontier_embds, cache_embds.unsqueeze(-1)).squeeze(-1)
        logits = nn.functional.softmax(torch.abs(logits) *  mask)

        frontier_embds = torch.sum(frontier_embds, dim=-2) /  torch.sum(mask, dim=-1).unsqueeze(-1)
        cache_embds = self.hidden_layer2(cache_embds)
        V = torch.bmm(frontier_embds.unsqueeze(1), cache_embds.unsqueeze(-1)).squeeze(-1)

        return logits, V

        



