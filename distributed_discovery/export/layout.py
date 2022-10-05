"""
    This file contains modified code from PM4Py.

    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
"""
from typing import List, Dict, Tuple
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import tempfile

from graphviz import Digraph
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.bpmn.util.sorting import get_sorted_nodes_edges
from pm4py.objects.bpmn.layout.variants.graphviz import (
    EndpointDirection,
    get_right_edge_coord,
    get_left_edge_coord,
    get_top_edge_coord,
    get_bottom_edge_coord,
)


@dataclass
class ClusterPosition:
    """
    Position of a participant lane in the graphviz Digraph.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


def initialize_endpoint() -> Dict[EndpointDirection, float]:
    """
    Initializes a default dictionary containing all `EndpointDirection`s.

    Returns
    -------
    dict
        Initialized dictionary.
    """
    return {
        EndpointDirection.RIGHT: 0.0,
        EndpointDirection.LEFT: 0.0,
        EndpointDirection.TOP: 0.0,
        EndpointDirection.BOTTOM: 0.0,
    }


def graphviz_layout(bpmn_graphs: List[BPMN]) -> Tuple[str, Dict[str, BPMN.BPMNNode]]:
    """
    Creates a graphviz Digraph for a collaboration model.
    Each node is drawn with the same shape to retrieve the coordinates of the nodes.
    Message flows are not drawn.

    Parameters
    ----------
    bpmn_graphs
        A list of BPMN graphs.

    Returns
    -------
    content
        The SVG source of the resulting graphviz diagram.
    graph_nodes_id
        A dictionary of the ids in the graphviz diagram with their BPMN node.

    """
    with tempfile.NamedTemporaryFile(
        suffix=".gv"
    ) as file_gv, tempfile.NamedTemporaryFile(suffix=".svg") as file_svg:
        parent_dot = Digraph(
            "parent",
            filename=file_gv.name,
            graph_attr={
                "rankdir": "LR",
            },
            engine="dot",
            format="svg",
        )

        graph_nodes_id: Dict[str, BPMN.BPMNNode] = {}

        for bpmn_graph in bpmn_graphs:
            subgraph = Digraph(
                f"cluster_{bpmn_graph.get_name()}",  # Name needs to start with cluster
                engine="dot",
                graph_attr={
                    "label": bpmn_graph.get_name(),
                    "rankdir": "LR",
                },
                node_attr={"shape": "box"},
            )

            nodes, edges = get_sorted_nodes_edges(bpmn_graph)

            for n in nodes:
                node_id = str(id(n))
                subgraph.node(
                    node_id,
                    shape="box",
                    label=n.get_name(),
                )
                graph_nodes_id[node_id] = n

            for e in edges:
                subgraph.edge(str(id(e[0])), str(id(e[1])))

            parent_dot.subgraph(subgraph)

        parent_dot.render(outfile=file_svg.name)
        content = file_svg.read().decode("utf-8")

        return content, graph_nodes_id


def layout_nodes(
    nodes: List[BPMN.BPMNNode],
    viz_nodes: List[str],
    graph_nodes_id: Dict[str, BPMN.BPMNNode],
    endpoints_wh: int,
    task_wh: int,
) -> None:
    """
    Retrieves the nodes coordinates in a SVG file.

    Parameters
    ----------
    nodes
        A list of all BPMN nodes.
    viz_nodes
        A list of all SVG nodes.
    graph_nodes_id
        A dictionary of the ids in the graphviz diagram with their BPMN node.
    endpoints_wh
        The default width and height for a start/end endpoint.
    task_wh
        The default width and height for a task.

    Returns
    -------

    """
    nodes_pos = {}

    for node in viz_nodes:
        this_id = node.split("<title>")[1].split("</title>")[0]
        points = node.split('points="')[1].split('"')[0]
        node_id = graph_nodes_id[this_id].get_id()
        nodes_pos[node_id] = points

    # add node positions to BPMN nodes
    for node in nodes:
        node_pos = nodes_pos[node.get_id()].split(" ")[0].split(",")

        pos_x = float(node_pos[0])
        pos_y = float(node_pos[1])
        node.set_x(pos_x)
        node.set_y(pos_y)
        if isinstance(node, BPMN.Task):
            this_width = min(
                round(2 * task_wh),
                round(2 * (len(node.get_name()) + 7) * task_wh / 22.0),
            )
            node.set_width(this_width)
            node.set_height(task_wh)
        elif isinstance(
            node,
            (
                BPMN.EndEvent,
                BPMN.MessageIntermediateCatchEvent,
                BPMN.MessageIntermediateThrowEvent,
                BPMN.StartEvent,
            ),
        ):
            node.set_width(endpoints_wh)
            node.set_height(endpoints_wh)
        else:
            node.set_width(task_wh)
            node.set_height(task_wh)

    max_x = max(1, max(abs(node.get_x()) for node in nodes))
    max_y = max(1, max(abs(node.get_y()) for node in nodes))

    stretch_fact_x = 1.25 * 1920.0 / max_x
    stretch_fact_y = 1080.0 / max_y

    for node in nodes:
        x = round(node.get_x() * stretch_fact_x)
        y = round(node.get_y() * stretch_fact_y)
        node.set_x(x)
        node.set_y(y)


def layout_clusters(
    bpmn_graphs: List[BPMN], task_wh: int
) -> Dict[str, ClusterPosition]:
    """
    Specifies the layout for every participant's lane.

    Parameters
    ----------
    bpmn_graphs
        List of all BPMN graphs.
    task_wh
        The default width of a task.

    Returns
    -------
    clusters_pos
        A dictionary of each participant with their `ClusterPosition`.

    """
    clusters_pos = {
        bpmn_graph.get_name(): ClusterPosition() for bpmn_graph in bpmn_graphs
    }

    for bpmn_graph in bpmn_graphs:
        x_coords = [node.get_x() for node in bpmn_graph.get_nodes()]
        y_coords = [node.get_y() for node in bpmn_graph.get_nodes()]

        min_x = min(x_coords)
        max_x = max(x_coords)

        min_y = min(y_coords)
        max_y = max(y_coords)

        cluster_position = clusters_pos[bpmn_graph.get_name()]

        cluster_position.x = min_x - task_wh
        cluster_position.y = min_y - task_wh

        cluster_position.height = abs(min_y - max_y) + 3 * task_wh
        cluster_position.width = abs(min_x - max_x) + 3 * task_wh

    return clusters_pos


def layout_flows(flows: List[BPMN.Flow]) -> None:
    """
    Specifies the layout of all flows in a participant.

    Parameters
    ----------
    flows
        A list of all flows.

    Returns
    -------

    """
    sources_dict, targets_dict, outgoing_edges, ingoing_edges = set_flow_coords(flows)

    # normalization
    normalize_edges(outgoing_edges, ingoing_edges)
    normalize_edges(ingoing_edges, outgoing_edges)

    # keep best direction
    for p1 in outgoing_edges:
        for p2 in outgoing_edges[p1]:
            vals = sorted(
                list(outgoing_edges[p1][p2].items()),
                key=lambda x: x[1],
                reverse=True,
            )
            outgoing_edges[p1][p2] = vals[0][0]
    for p1 in ingoing_edges:
        for p2 in ingoing_edges[p1]:
            vals = sorted(
                list(ingoing_edges[p1][p2].items()),
                key=lambda x: x[1],
                reverse=True,
            )
            ingoing_edges[p1][p2] = vals[0][0]

    total_counter = create_flow_counter(outgoing_edges, ingoing_edges)
    partial_counter = {}

    outgoing_edges_dirs = deepcopy(outgoing_edges)
    ingoing_edges_dirs = deepcopy(ingoing_edges)

    # decide exiting/entering point for edges
    decide_edge_origin_target(
        outgoing_edges, sources_dict, partial_counter, total_counter
    )
    decide_edge_origin_target(
        ingoing_edges, targets_dict, partial_counter, total_counter
    )

    # order the left-entering ingoing edges better
    for p1 in ingoing_edges:
        vals = [(x, y) for x, y in ingoing_edges[p1].items() if y[0] == p1[0]]
        if len(vals) > 1:
            vals_x = [x[0] for x in vals]
            vals_y = [x[1] for x in vals]
            vals_x = sorted(vals_x)
            vals_y = sorted(vals_y)
            for i in range(len(vals_x)):
                ingoing_edges[p1][vals_x[i]] = vals_y[i]

    # set waypoints for edges
    set_flow_waypoints(
        flows, outgoing_edges, ingoing_edges, outgoing_edges_dirs, ingoing_edges_dirs
    )


def create_flow_counter(
    outgoing_edges: Dict, ingoing_edges: Dict
) -> Dict[str, Counter[EndpointDirection, int]]:
    """
    Creates a counter for all flows and its `EndpointDirection`s.

    Parameters
    ----------
    outgoing_edges
        A 2D-Dictionary containing `EndpointDirection`s.
    ingoing_edges
        A 2D-Dictionary containing `EndpointDirection`s.
    Returns
    -------
    total_counter
        A dictionary containing a coordinate with a Counter of `EndpointDirection`s.

    """
    total_counter = {}

    for p1 in outgoing_edges:
        if p1 not in total_counter:
            total_counter[p1] = Counter()
        for p2 in outgoing_edges[p1]:
            direction = outgoing_edges[p1][p2]
            total_counter[p1][direction] += 1
    for p1 in ingoing_edges:
        if p1 not in total_counter:
            total_counter[p1] = Counter()
        for p2 in ingoing_edges[p1]:
            direction = ingoing_edges[p1][p2]
            total_counter[p1][direction] += 1

    return total_counter


def set_flow_coords(
    flows: List[BPMN.Flow],
) -> Tuple[
    Dict[Tuple[float, float], BPMN.BPMNNode],
    Dict[Tuple[float, float], BPMN.BPMNNode],
    Dict,
    Dict,
]:
    """
    Sets initial coordinates for all flows and also initializes dictionaries of target and source flows.

    Parameters
    ----------
    flows
        List of all flows.

    Returns
    -------
    sources_dict
        Dictionary of coordinates with the respective source node.
    targets_dict
        Dictionary of coordinates with the respective target node.
    outgoing_edges
        A 2D-Dictionary containing `EndpointDirection`s.
    ingoing_edges
        A 2D-Dictionary containing `EndpointDirection`s.

    """
    outgoing_edges = {}
    ingoing_edges = {}
    sources_dict: Dict[Tuple[float, float], BPMN.BPMNNode] = {}
    targets_dict: Dict[Tuple[float, float], BPMN.BPMNNode] = {}

    for flow in flows:
        source = flow.get_source()
        target = flow.get_target()

        x_src = source.get_x()
        x_trg = target.get_x()
        y_src = source.get_y()
        y_trg = target.get_y()

        sources_dict[(x_src, y_src)] = source
        targets_dict[(x_trg, y_trg)] = target

        diff_x = abs(x_trg - x_src)
        diff_y = abs(y_src - y_trg)

        if (x_src, y_src) not in outgoing_edges:
            outgoing_edges[(x_src, y_src)] = {}

        outgoing_edges[(x_src, y_src)][(x_trg, y_trg)] = initialize_endpoint()

        if (x_trg, y_trg) not in ingoing_edges:
            ingoing_edges[(x_trg, y_trg)] = {}

        ingoing_edges[(x_trg, y_trg)][(x_src, y_src)] = initialize_endpoint()

        outgoing_edge_direction = EndpointDirection.LEFT
        ingoing_edge_direction = EndpointDirection.RIGHT

        if x_trg > x_src:
            outgoing_edge_direction = EndpointDirection.RIGHT
            ingoing_edge_direction = EndpointDirection.LEFT

        outgoing_edges[(x_src, y_src)][(x_trg, y_trg)][
            outgoing_edge_direction
        ] = diff_x / (diff_x + diff_y)
        ingoing_edges[(x_trg, y_trg)][(x_src, y_src)][
            ingoing_edge_direction
        ] = diff_x / (diff_x + diff_y)

        outgoing_edge_direction = EndpointDirection.BOTTOM
        ingoing_edge_direction = EndpointDirection.TOP

        if y_src > y_trg:
            outgoing_edge_direction = EndpointDirection.TOP
            ingoing_edge_direction = EndpointDirection.BOTTOM

        outgoing_edges[(x_src, y_src)][(x_trg, y_trg)][
            outgoing_edge_direction
        ] = diff_y / (diff_x + diff_y)
        ingoing_edges[(x_trg, y_trg)][(x_src, y_src)][
            ingoing_edge_direction
        ] = diff_y / (diff_x + diff_y)

    return sources_dict, targets_dict, outgoing_edges, ingoing_edges


def set_flow_waypoints(
    flows: List[BPMN.Flow],
    outgoing_edges: Dict,
    ingoing_edges: Dict,
    outgoing_edges_dirs: Dict,
    ingoing_edges_dirs: Dict,
):
    """
    Sets waypoints of flows.

    Parameters
    ----------
    flows
        List of all flows.
    outgoing_edges
        A 2D-Dictionary for outgoing edges.
    ingoing_edges
        A 2D-Dictionary for ingoing edges.
    outgoing_edges_dirs
        A 2D-Dictionary for outgoing edges.
    ingoing_edges_dirs
        A 2D-Dictionary for ingoing edges.
    Returns
    -------

    """
    for flow in flows:
        source = flow.get_source()
        target = flow.get_target()

        flow.del_waypoints()

        x_src = source.get_x()
        x_trg = target.get_x()
        y_src = source.get_y()
        y_trg = target.get_y()
        p1 = (x_src, y_src)
        p2 = (x_trg, y_trg)

        source_x = outgoing_edges[p1][p2][0]
        source_y = outgoing_edges[p1][p2][1]
        target_x = ingoing_edges[p2][p1][0]
        target_y = ingoing_edges[p2][p1][1]
        dir_source = outgoing_edges_dirs[p1][p2]
        dir_target = ingoing_edges_dirs[p2][p1]

        middle_x = (source_x + target_x) / 2.0
        middle_y = (source_y + target_y) / 2.0

        flow.add_waypoint((source_x, source_y))
        if dir_source in [EndpointDirection.LEFT, EndpointDirection.RIGHT]:
            if dir_target in [EndpointDirection.LEFT, EndpointDirection.RIGHT]:
                flow.add_waypoint((middle_x, source_y))
                flow.add_waypoint((middle_x, target_y))
            elif dir_target in [EndpointDirection.TOP, EndpointDirection.BOTTOM]:
                flow.add_waypoint((target_x, source_y))
        elif dir_source in [EndpointDirection.TOP, EndpointDirection.BOTTOM]:
            if dir_target in [EndpointDirection.TOP, EndpointDirection.BOTTOM]:
                flow.add_waypoint((source_x, middle_y))
                flow.add_waypoint((target_x, middle_y))
            elif dir_target in [EndpointDirection.LEFT, EndpointDirection.RIGHT]:
                flow.add_waypoint((source_x, target_y))

        flow.add_waypoint((target_x, target_y))


def normalize_edges(source_edges: Dict, target_edges: Dict) -> None:
    """
    Normalizes edges and changes the input dictionaries type in-place.

    Parameters
    ----------
    source_edges
        A 2D-Dictionary containing `EndpointDirection`s.
    target_edges
        A 2D-Dictionary containing `EndpointDirection`s.

    Returns
    -------

    """
    source_edges0 = deepcopy(source_edges)
    target_edges0 = deepcopy(target_edges)

    for p1 in source_edges:
        sum_right = 0.0
        sum_left = 0.0
        sum_top = 0.0
        sum_bottom = 0.0
        for p2 in source_edges[p1]:
            sum_right += source_edges0[p1][p2][EndpointDirection.RIGHT]
            sum_left += source_edges0[p1][p2][EndpointDirection.LEFT]
            sum_top += source_edges0[p1][p2][EndpointDirection.TOP]
            sum_bottom += source_edges0[p1][p2][EndpointDirection.BOTTOM]
        if p1 in target_edges:
            for p2 in target_edges[p1]:
                sum_right += target_edges0[p1][p2][EndpointDirection.RIGHT]
                sum_left += target_edges0[p1][p2][EndpointDirection.LEFT]
                sum_top += target_edges0[p1][p2][EndpointDirection.TOP]
                sum_bottom += target_edges0[p1][p2][EndpointDirection.BOTTOM]
        for p2 in source_edges[p1]:
            if sum_right > 0:
                source_edges[p1][p2][EndpointDirection.RIGHT] = (
                    source_edges[p1][p2][EndpointDirection.RIGHT] ** 2 / sum_right
                )
            if sum_left > 0:
                source_edges[p1][p2][EndpointDirection.LEFT] = (
                    source_edges[p1][p2][EndpointDirection.LEFT] ** 2 / sum_left
                )
            if sum_top > 0:
                source_edges[p1][p2][EndpointDirection.TOP] = (
                    source_edges[p1][p2][EndpointDirection.TOP] ** 2 / sum_top
                )
            if sum_bottom > 0:
                source_edges[p1][p2][EndpointDirection.BOTTOM] = (
                    source_edges[p1][p2][EndpointDirection.BOTTOM] ** 2 / sum_bottom
                )


def decide_edge_origin_target(edges, nodes, partial_counter, total_counter) -> None:
    """
    Decides exiting/entering point for edges.

    Parameters
    ----------
    edges
        A 2D-Dictionary containing `EndpointDirection`s.
    nodes
        Dictionary of coordinates with the respective node.
    partial_counter
        A dictionary containing a coordinate with a Counter of `EndpointDirection`s.
    total_counter
        A dictionary containing a coordinate with a Counter of `EndpointDirection`s.

    Returns
    -------

    """
    for p1 in edges:
        node = nodes[p1]
        if p1 not in partial_counter:
            partial_counter[p1] = Counter()
        sorted_edges = sorted(edges[p1], key=lambda x: x, reverse=False)
        for p2 in sorted_edges:
            direction = edges[p1][p2]
            partial_counter[p1][direction] += 1
            if direction == EndpointDirection.RIGHT:
                edges[p1][p2] = get_right_edge_coord(
                    node, p1, partial_counter[p1], total_counter[p1]
                )
            elif direction == EndpointDirection.LEFT:
                edges[p1][p2] = get_left_edge_coord(
                    node, p1, partial_counter[p1], total_counter[p1]
                )
            elif direction == EndpointDirection.TOP:
                edges[p1][p2] = get_top_edge_coord(
                    node, p1, partial_counter[p1], total_counter[p1]
                )
            elif direction == EndpointDirection.BOTTOM:
                edges[p1][p2] = get_bottom_edge_coord(
                    node, p1, partial_counter[p1], total_counter[p1]
                )


def layout_bpmn_graphs(
    bpmn_graphs: List[BPMN], endpoints_wh: int = 30, task_wh: int = 60
) -> Dict[str, ClusterPosition]:
    """
    Sets for every element in collaboration BPMN model the coordinates and waypoints.

    Parameters
    ----------
    bpmn_graphs
        A list of BPMN models.
    endpoints_wh
        The default width and height for a start/end endpoint.
    task_wh
        The default width and height for a task.

    Returns
    -------
    clusters_pos
        A dictionary of each participant with their `ClusterPosition`.
    """
    content, graph_nodes_id = graphviz_layout(bpmn_graphs)

    nodes = [node for bpmn_graph in bpmn_graphs for node in bpmn_graph.get_nodes()]
    flows = [flow for bpmn_graph in bpmn_graphs for flow in bpmn_graph.get_flows()]

    viz_nodes = content.split('class="node">')[1:]

    layout_nodes(nodes, viz_nodes, graph_nodes_id, endpoints_wh, task_wh)
    clusters_pos = layout_clusters(bpmn_graphs, task_wh)
    layout_flows(flows)

    return clusters_pos
