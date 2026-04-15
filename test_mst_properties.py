"""
E0 251o (2026) - Property-Based Testing for NetworkX

Team members:
    Rahul Rai (SR No. 24765)

Algorithms under test:
    Minimum Spanning Tree - tested via nx.minimum_spanning_tree (undirected, weighted)

This module tests:
    - A subset property on connected random graphs (weight sum).
    - The strict cycle property together with empty / singleton / disconnected /
      self-loop edge cases, all exercised inside
      test_cycle_property_unique_heaviest_edge_not_in_mst.

Graph generation uses connected Erdos-Renyi graphs (2-20 vertices) for the weight-sum property.
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


@st.composite
def disconnected_weighted_union_graphs(draw):
    """
    Two to four pairwise disconnected components, each connected internally, weighted.
    """
    num_components = draw(st.integers(min_value=2, max_value=4))
    combined = nx.Graph()
    next_label = 0
    for _ in range(num_components):
        component_order = draw(st.integers(min_value=2, max_value=8))
        component_p = draw(
            st.floats(
                min_value=0.4,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            )
        )
        component_seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        component = nx.gnp_random_graph(
            component_order, component_p, seed=component_seed
        )
        assume(nx.is_connected(component))
        mapping = {i: next_label + i for i in range(component_order)}
        component = nx.relabel_nodes(component, mapping)
        for u, v in component.edges():
            component[u][v]["weight"] = draw(st.integers(min_value=1, max_value=50))
        combined.update(component)
        next_label += component_order
    return combined


@st.composite
def small_connected_weighted_undirected_graphs(draw):
    """
    Connected weighted graphs with at most 12 vertices — used where enumerating
    nx.simple_cycles must stay fast (e.g. connected graph plus self-loop).
    """
    num_vertices = draw(st.integers(min_value=2, max_value=12))
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
def small_connected_weighted_graph_with_self_loop(draw):
    """
    Small connected weighted graph plus one weighted self-loop (keeps cycle search tractable).
    """
    base_graph = draw(small_connected_weighted_undirected_graphs())
    graph_with_loop = base_graph.copy()
    loop_vertex = draw(
        st.integers(min_value=0, max_value=graph_with_loop.number_of_nodes() - 1)
    )
    loop_weight = draw(st.integers(min_value=1, max_value=100))
    graph_with_loop.add_edge(loop_vertex, loop_vertex, weight=loop_weight)
    return graph_with_loop


@st.composite
def graphs_for_cycle_property_including_edge_cases(draw):
    """
    Mix of edge-case graphs and random connected graphs so one test can cover:
    empty graph; single vertex (plain or with self-loop); disconnected weighted union;
    connected graphs suitable for simple-cycle enumeration; connected graphs with a
    self-loop attached.
    """
    kind = draw(
        st.sampled_from(
            [
                "empty",
                "singleton",
                "singleton_self_loop",
                "disconnected",
                "connected_for_cycles",
                "connected_with_self_loop",
            ]
        )
    )
    if kind == "empty":
        return nx.Graph()
    if kind == "singleton":
        graph = nx.Graph()
        graph.add_node(0)
        return graph
    if kind == "singleton_self_loop":
        graph = nx.Graph()
        graph.add_node(0)
        graph.add_edge(0, 0, weight=draw(st.integers(min_value=1, max_value=100)))
        return graph
    if kind == "disconnected":
        return draw(disconnected_weighted_union_graphs())
    if kind == "connected_for_cycles":
        return draw(small_connected_weighted_graphs_for_cycle_property())
    return draw(small_connected_weighted_graph_with_self_loop())


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


@given(graphs_for_cycle_property_including_edge_cases())
@settings(max_examples=30, deadline=None)
def test_cycle_property_unique_heaviest_edge_not_in_mst(weighted_input_graph):
    """
    Property: For any simple cycle C in a graph, if one edge e has a strictly greater 
    weight than all other edges in C, then e is not in the minimum spanning tree.

    This same test also checks edge cases that interact with spanning forests:

    - Empty graph: MST has no vertices and no edges.
    - Single vertex (no self-loop): MST has one vertex and no edges.
    - Single vertex with a self-loop: MST still has no edges; self-loops never appear
      in the MST edge set.
    - Disconnected graph (c ≥ 2 components): MST edge count is n - c (spanning forest).
    - Any graph: MST edges are never self-loops (u, u).

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
    graph = weighted_input_graph
    num_vertices = graph.number_of_nodes()
    num_components = nx.number_connected_components(graph)

    minimum_spanning_tree = nx.minimum_spanning_tree(graph)

    for u, v in minimum_spanning_tree.edges():
        assert u != v, "MST must not contain self-loop edges"

    if num_vertices == 0:
        assert minimum_spanning_tree.number_of_nodes() == 0
        assert minimum_spanning_tree.number_of_edges() == 0
        return

    if num_vertices == 1:
        assert minimum_spanning_tree.number_of_nodes() == 1
        assert minimum_spanning_tree.number_of_edges() == 0
        return

    assert minimum_spanning_tree.number_of_edges() == num_vertices - num_components

    mst_edge_set = {
        _normalized_undirected_edge(u, v) for u, v in minimum_spanning_tree.edges()
    }

    for cycle_vertices in nx.simple_cycles(graph, length_bound=maximum_cycle_length):
        num_cycle_vertices = len(cycle_vertices)
        cycle_edges_with_weights = []
        for index in range(num_cycle_vertices):
            u = cycle_vertices[index]
            v = cycle_vertices[(index + 1) % num_cycle_vertices]
            edge_key = _normalized_undirected_edge(u, v)
            cycle_edges_with_weights.append(
                (edge_key, _edge_weight(graph, u, v))
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
