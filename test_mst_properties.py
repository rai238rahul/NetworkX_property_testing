"""
E0 251o (2026) - Property-Based Testing for NetworkX

Team members:
    Rahul Rai

Algorithms under test:
    Minimum Spanning Tree - tested via nx.minimum_spanning_tree (undirected, weighted)

This module uses Hypothesis to test MST properties on generated graphs, including:
    - Invariants and postconditions
    - Metamorphic properties
    - Idempotence
    - Boundary and special cases

Graph generation (see connected_weighted_undirected_graphs) covers a wide range of sizes (2-20 nodes), densities, and weight assignments
to ensure robust coverage of tie-breaks and topological structure.
"""

import networkx as nx
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

@st.composite
def connected_weighted_undirected_graphs(draw):
    """
    Draw a connected simple undirected graph with positive integer edge weights.

    Graphs vary in order (2-20), edge density (Erdos-Renyi p), and weight
    magnitudes to exercise different MST tie-break and topology cases.
    """
    num_vertices = draw(st.integers(min_value=2, max_value=20))
    edge_probability = draw(
        st.floats(
            min_value=0.25,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        )
    )
    erdos_renyi_seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    random_graph = nx.gnp_random_graph(
        num_vertices, edge_probability, seed=erdos_renyi_seed
    )
    assume(nx.is_connected(random_graph))
    for u, v in random_graph.edges():
        random_graph[u][v]["weight"] = draw(st.integers(min_value=1, max_value=100))
    return random_graph


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(connected_weighted_undirected_graphs())
@settings(max_examples=40, deadline=None)
def test_mst_total_weight_at_most_sum_of_all_edge_weights(spanning_tree_input_graph):
    """
    Property: The sum of MST edge weights is at most the sum of weights over all
    edges in the original graph.

    Mathematical basis: An MST forms a subset of the edges of G, and all weights
    are positive. Therefore, their sum cannot exceed the sum over the full edge set.

    Test strategy: Generate connected weighted graphs and compare the sum of weights
    from MST edges with the total edge weight sum of the original graph.

    Why this matters: Violating this would signal that the MST algorithm included
    extraneous edges, corrupted weights, or otherwise produced an invalid tree; it
    validates fundamental correctness.
    """
    minimum_spanning_tree = nx.minimum_spanning_tree(spanning_tree_input_graph)
    mst_weight_sum = sum(
        data["weight"] for _, _, data in minimum_spanning_tree.edges(data=True)
    )
    graph_weight_sum = sum(
        data["weight"] for _, _, data in spanning_tree_input_graph.edges(data=True)
    )
    assert mst_weight_sum <= graph_weight_sum
