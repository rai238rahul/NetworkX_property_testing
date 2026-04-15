"""
E0 251o (2026) - Property-Based Testing for NetworkX

Team members:
    Rahul Rai (SR No. 24765)

Algorithms under test:
    Minimum Spanning Tree - tested via nx.minimum_spanning_tree (undirected, weighted)

This module currently tests:
    - A subset property: MST total weight does not exceed the sum of all edge weights.
    - The cycle property: on any simple cycle, an edge that is uniquely maximum weight
      on that cycle is not in the MST.

Graph generation uses connected Erdos-Renyi graphs (2-20 vertices) for the weight-sum property,
and smaller graphs (4-10 vertices) for cycle enumeration, since listing simple cycles is costly
on large dense instances.
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


@st.composite
def small_connected_weighted_graphs_for_cycle_property(draw):
    """
    Connected weighted graphs with modest order (4-10 vertices) for cycle enumeration.

    Simple-cycle listing via nx.simple_cycles grows quickly with density and n; this
    strategy keeps examples tractable while still including triangles, longer cycles,
    and chorded structures.
    """
    num_vertices = draw(st.integers(min_value=4, max_value=10))
    edge_probability = draw(
        st.floats(
            min_value=0.35,
            max_value=0.95,
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
        random_graph[u][v]["weight"] = draw(st.integers(min_value=1, max_value=50))
    return random_graph


def _normalized_undirected_edge(first_endpoint, second_endpoint):
    """
    Normalize an undirected edge to a canonical form (u, v) where u <= v.
    """
    if first_endpoint <= second_endpoint:
        return (first_endpoint, second_endpoint)
    return (second_endpoint, first_endpoint)


def _edge_weight(weighted_graph, first_endpoint, second_endpoint):
    """
    Get the weight of an edge in a weighted graph.
    """
    a, b = _normalized_undirected_edge(first_endpoint, second_endpoint)
    return weighted_graph[a][b]["weight"]


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

    Assumptions: Undirected simple graph; one connected component; every edge has a
    numeric 'weight' attribute ≥ 1.

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


@given(small_connected_weighted_graphs_for_cycle_property())
@settings(max_examples=25, deadline=None)
def test_cycle_property_unique_heaviest_edge_not_in_mst(weighted_input_graph):
    """
    Property: For any simple cycle C in a graph, if one edge e has a strictly greater 
    weight than all other edges in C, then e is not in the minimum spanning tree.

    Mathematical basis: If a spanning tree T contained e, removing e splits T into two
    components. The path C \ {e} connects the endpoints of e using only edges lighter
    than e, so some lighter edge crosses the same cut and can replace e, lowering total
    weight - contradicting minimality.

    Tie-breaking note: If several edges on C tie for maximum weight, an MST may
    include one of them; this test only asserts the case of a unique maximum on C.

    Test strategy: For each simple cycle from nx.simple_cycles (length cap 12), if
    exactly one edge attains the cycle maximum weight, assert it is absent from the MST
    edge set.

    Assumptions: Undirected simple graph; positive integer weights; connected input
    (see small_connected_weighted_graphs_for_cycle_property).

    Why this matters: Violating this would signal that the MST algorithm included
    extraneous edges, corrupted weights, or otherwise produced an invalid tree; it
    validates fundamental correctness.
    """
    maximum_cycle_length = 12
    minimum_spanning_tree = nx.minimum_spanning_tree(weighted_input_graph)
    mst_edge_set = {
        _normalized_undirected_edge(u, v) for u, v in minimum_spanning_tree.edges()
    }

    for cycle_vertices in nx.simple_cycles(
        weighted_input_graph, length_bound=maximum_cycle_length
    ):
        num_cycle_vertices = len(cycle_vertices)
        cycle_edges_with_weights = []
        for index in range(num_cycle_vertices):
            u = cycle_vertices[index]
            v = cycle_vertices[(index + 1) % num_cycle_vertices]
            edge_key = _normalized_undirected_edge(u, v)
            cycle_edges_with_weights.append(
                (edge_key, _edge_weight(weighted_input_graph, u, v))
            )
        maximum_weight_on_cycle = max(weight for _, weight in cycle_edges_with_weights)
        edges_at_maximum = [
            edge_key
            for edge_key, weight in cycle_edges_with_weights
            if weight == maximum_weight_on_cycle
        ]
        if len(edges_at_maximum) == 1:
            uniquely_heaviest_edge = edges_at_maximum[0]
            assert uniquely_heaviest_edge not in mst_edge_set
