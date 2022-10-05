"""
    This file contains modified code from PM4Py
    to visualize BPMN collaboration diagrams.

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
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import List, Dict, Set, Optional

from graphviz import Digraph
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.bpmn.util.sorting import get_sorted_nodes_edges

from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.util.bpmn import copy_message_bpmn


def visualize_bpmn(
    bpmn_graphs: List[BPMN],
    message_bpmn: Dict[MessageFlow, List[BPMN.BPMNNode]],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
    image_format: str = "png",
    background_color: str = "white",
    font_size: int = 15,
    layout: str = "LR",
    show_message_label: bool = True,
    color_start_end_events: bool = True,
    assets_dir: Optional[str] = None,
) -> Digraph:
    """
    Visualizes a list of BPMN graphs as clusters in a Graphviz Digraph.

    Parameters
    ----------
    bpmn_graphs:
        List of BPMN graphs. One per participant.
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.
    sent_messages
        A dictionary of message flows per participant.
    image_format
        Format of the resulting image. Default: PNG.
    background_color
        Background color of the graph. Default: white.
    font_size
        Font size used in the graph. Default: 12.
    layout
        The layout of the graph. Default: Left-to-right.
    show_message_label
        Displays an edge label for message activities.
    color_start_end_events
        If set draws start events in green and end events in orange.
    assets_dir
        Optionally specify location of assets folder

    Returns
    -------
    dot
        A single Graphviz Digraph object containing all BPMN graphs as clusters.
    """
    font_size = str(font_size)

    with NamedTemporaryFile(suffix=".gv") as file:

        parent_graph = Digraph(
            name="parent",
            filename=file.name,
            graph_attr={
                "bgcolor": background_color,
                "rankdir": layout,
                "fontsize": font_size,
            },
            format=image_format,
        )

        # Absolute location of assets folder
        assets_dir = (
            f"{Path(__file__).parent.resolve()}/assets"
            if assets_dir is None
            else assets_dir
        )

        for bpmn_graph in bpmn_graphs:
            cluster_viz = bpmn_to_graphviz(
                bpmn_graph,
                assets_dir,
                font_size,
                background_color,
                layout,
                color_start_end_events,
            )
            parent_graph.subgraph(cluster_viz)

            del cluster_viz

        visualize_message_flows(
            parent_graph, sent_messages, message_bpmn, show_message_label, font_size
        )

        parent_graph.attr(overlap="false")

        return parent_graph


def visualize_message_flows(
    dot: Digraph, sent_messages, message_bpmn, show_message_label, font_size: str
):
    """
    Visualizes message flows in a BPMN Digraph.

    Parameters
    ----------
    dot
        The Digraph object.
    sent_messages
        A dictionary of message flows per participant.
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.
    show_message_label
        Displays an edge label for message activities.
    font_size
        The font size to use.

    Returns
    -------

    """
    message_bpmn = copy_message_bpmn(message_bpmn)

    for participant, entries in sent_messages.items():
        for sender, receivers in entries.items():
            sender_bpmn_nodes = message_bpmn[MessageFlow(participant, sender)]
            receiver_tuples = [
                (node, receiver)
                for receiver in receivers
                for node in message_bpmn[receiver]
            ]

            for sender_bpmn_node, receiver_tuple in zip(
                sender_bpmn_nodes, receiver_tuples
            ):
                receiver_bpmn_node, receiver = receiver_tuple
                # Removes entry from list for correct connection to receiver task
                message_bpmn[receiver].pop(0)

                tail_label = ""
                head_label = ""

                if show_message_label:
                    # Adding white-space improves readability
                    tail_label = f"   {sender_bpmn_node.get_name()}   "
                    head_label = f"   {receiver_bpmn_node.get_name()}   "

                dot.edge(
                    str(id(sender_bpmn_node)),
                    str(id(receiver_bpmn_node)),
                    taillabel=tail_label,
                    headlabel=head_label,
                    _attributes={
                        "style": "dotted",
                        "arrowtail": "odot",
                        "dir": "both",
                    },
                    fontsize=font_size,
                )


def visualize_nodes(
    dot: Digraph, nodes, assets_dir: str, font_size: str, color_start_end_events: bool
) -> None:
    """
    Visualizes the nodes in a BPMN graph.

    Parameters
    ----------
    dot
        The Digraph object.
    nodes
        List of nodes in the BPMN graph.
    assets_dir
        Location of assets.
    font_size
        The font size of nodes.
    color_start_end_events
        If set draws start events in green and end events in orange.

    Returns
    -------

    """
    for n in nodes:
        n_id = str(id(n))
        if isinstance(n, BPMN.Task):
            dot.node(
                n_id,
                shape="box",
                label=n.get_name(),
                tooltip=n.get_name(),
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.MessageStartEvent):
            dot.node(
                n_id,
                shape="circle",
                image=f"{assets_dir}/envelope.svg",
                label="",
                style="filled",
                fillcolor="green" if color_start_end_events else "transparent",
                tooltip=n.get_name(),
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.StartEvent):
            dot.node(
                n_id,
                label="",
                shape="circle",
                style="filled",
                fillcolor="green" if color_start_end_events else "transparent",
                tooltip="Start Event",
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.MessageEndEvent):
            dot.node(
                n_id,
                shape="circle",
                image=f"{assets_dir}/envelope_full.svg",
                label="",
                style="filled",
                fillcolor="orange" if color_start_end_events else "transparent",
                tooltip=n.get_name(),
                penwidth="3",
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.EndEvent):
            dot.node(
                n_id,
                label="",
                shape="circle",
                style="filled",
                fillcolor="orange" if color_start_end_events else "transparent",
                tooltip="End Event",
                penwidth="3",
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.ParallelGateway):
            dot.node(
                n_id,
                label="+",
                shape="diamond",
                tooltip="Parallel Gateway",
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.ExclusiveGateway):
            dot.node(
                n_id,
                label="X",
                shape="diamond",
                tooltip="Exclusive Gateway",
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.InclusiveGateway):
            dot.node(
                n_id,
                label="O",
                shape="diamond",
                tooltip="Inclusive Gateway",
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.MessageIntermediateCatchEvent):
            dot.node(
                n_id,
                shape="doublecircle",
                image=f"{assets_dir}/envelope.svg",
                label="",
                tooltip=n.get_name(),
                fontsize=font_size,
            )
        elif isinstance(n, BPMN.MessageIntermediateThrowEvent):
            dot.node(
                n_id,
                shape="doublecircle",
                image=f"{assets_dir}/envelope_full.svg",
                label="",
                tooltip=n.get_name(),
                fontsize=font_size,
            )
        else:
            dot.node(n_id, label="", shape="circle", fontsize=font_size)


def bpmn_to_graphviz(
    bpmn_graph: BPMN,
    assets_dir: str,
    font_size: str,
    background_color: str,
    layout: str,
    color_start_end_events: bool,
) -> Digraph:
    """
    Visualizes a BPMN Graph as a cluster.

    Parameters
    ----------
    bpmn_graph
        BPMN graph.
    assets_dir
        Absolute path of the assets folder.
    font_size
        Font size used in the graph.
    background_color
        Background color of the graph.
    layout
        The layout of the graph.
    color_start_end_events
        If set draws start events in green and end events in orange.

    Returns
    -------
    dot
        A Graphviz Digraph object.
    """
    if not isinstance(bpmn_graph, BPMN):
        raise TypeError("the method can be applied only to a 'BPMN'!")

    with NamedTemporaryFile(suffix=".gv") as file:
        name = bpmn_graph.get_name()

        dot = Digraph(
            f"cluster_{name}",  # Name needs to start with cluster
            filename=file.name,
            engine="dot",
            graph_attr={
                "bgcolor": background_color,
                "label": name,
                "tooltip": name,
                "rankdir": layout,
            },
            node_attr={"shape": "box"},
        )

        nodes, edges = get_sorted_nodes_edges(bpmn_graph)

        visualize_nodes(dot, nodes, assets_dir, font_size, color_start_end_events)

        for e in edges:
            n_id_1 = str(id(e[0]))
            n_id_2 = str(id(e[1]))

            dot.edge(n_id_1, n_id_2)

        return dot
