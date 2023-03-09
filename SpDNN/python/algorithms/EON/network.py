import networkx as nx
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import torch
from torch import nn

class EdgeOrderingNetwork(nn.Module):
    def __init__(self, graph, embedding_dim=8, hidden_dim=64):
        super(EdgeOrderingNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding.from_pretrained(self.node_emebddings(graph, embedding_dim))
        self.f_sentinel = nn.Parameter(torch.zeros(2*embedding_dim))
        self.c_sentinel = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.uniform_(self.f_sentinel, -1.0, 1.0)
        nn.init.uniform_(self.c_sentinel, -1.0, 1.0)

        self.Layer1 = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.self_attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

        self.FrontierLayer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.CacheLayer2 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

        self.softmax = nn.Softmax(dim=1)

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
        
        B, F, _ = frontier.size() # B * F * 2

        cache_embds = self.embedding(cache + 1) # B * C * E
        frontier_embds = self.embedding(frontier + 1) # B * F * 2 * E
        frontier_embds = torch.reshape( # B * F * (2 * E)
            frontier_embds, 
            (frontier_embds.shape[0], frontier_embds.shape[1], -1)
        ) 
        sentinel = self.f_sentinel.expand((B, 1, -1))
        frontier_embds = torch.cat((sentinel, frontier_embds), dim=1)
        sentinel = self.c_sentinel.expand((B, 1, -1))
        cache_embds = torch.cat((sentinel, cache_embds), dim=1)

        zeros = torch.zeros(B,1)
        frontier_mask = (torch.cat((zeros, frontier[:,:,0]), dim=1) < 0) # B * F
        cache_mask = (torch.cat((zeros, cache), dim=1) < 0)

        frontier_embds = self.Layer1(frontier_embds)
        frontier_embds = self.self_attention(
            frontier_embds, 
            frontier_embds, 
            frontier_embds,
            key_padding_mask=frontier_mask,
            need_weights=False
        )[0]

        frontier_embds = self.FrontierLayer2(frontier_embds)
        cache_embds = self.CacheLayer2(cache_embds)
        _, attn_weights = self.cross_attention(
            frontier_embds,
            cache_embds,
            cache_embds,
            key_padding_mask=cache_mask
        )
        logits = attn_weights[:, 1:, :].max(dim=2)[0]
        v = attn_weights[:, 1, :].sum(dim=1)
        logits = self.softmax(logits)
        return logits, v