import torch
import networkx as nx
from torch_geometric.utils import to_networkx


def compute_dist_and_norm(data) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (distance_matrix, normalisation_matrix) for a PyG Data graph."""
    def norm_from_dist(dist: torch.Tensor) -> torch.Tensor:
        N = dist.size(0)
        norm = torch.zeros_like(dist)
        for i in range(N):
            row = dist[i]
            # Consider only *finite* distances when counting
            finite_mask = torch.isfinite(row)
            counts = torch.bincount(row[finite_mask].long(
            )) if finite_mask.any() else torch.tensor([], dtype=torch.long)

            for j in range(N):
                if not torch.isfinite(row[j]):
                    # No path ⇒ normalisation of 1 to avoid division by zero
                    norm[i, j] = 1.0
                else:
                    d = int(row[j].item())
                    norm[i, j] = counts[d] if d < len(counts) else 1.0
        # Safety: ensure no zeros
        norm[norm == 0] = 1.0
        return norm

    g = to_networkx(data, to_undirected=True)
    sp = dict(nx.all_pairs_shortest_path_length(g))

    N = data.num_nodes
    # Initialise with +inf to mark "no path" entries explicitly
    dist = torch.full((N, N), float('inf'), dtype=torch.float)

    # Distance from each node to itself is 0 by definition
    dist.fill_diagonal_(0.0)

    # Fill finite shortest-path lengths returned by NetworkX
    for i, lengths in sp.items():
        for j, d in lengths.items():
            dist[int(i), int(j)] = float(d)

    # Compute the normalisation matrix; unreachable pairs (inf) get count 1
    norm = norm_from_dist(dist)
    return dist, norm


class PreprocessDistances:
    """PyG Transform that adds GNAN distance attributes to each graph."""
    def __call__(self, data):  # noqa: D401
        dist, norm = compute_dist_and_norm(data)
        data.node_distances = dist
        data.normalization_matrix = norm
        return data
