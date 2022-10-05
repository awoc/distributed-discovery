"""
    This file contains modified code from PM4Py
    to visualize process trees for distributed processes.

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
from copy import deepcopy
from typing import Optional
from graphviz import Graph

from pm4py.objects.process_tree.utils import generic

from distributed_discovery.objects.process_tree import ProcessTree
from distributed_discovery.objects.process_tree_operator import Operator


def visualize_process_tree(
    process_tree: ProcessTree,
    image_format: Optional[str] = "png",
    background_color: Optional[str] = "white",
    font_size: Optional[int] = 15,
    layout: Optional[str] = "LR",
) -> Graph:
    """
    Visualizes a process tree as Graphviz Graph.

    Parameters
    ----------
    process_tree
        Process Tree.
    image_format
        Format of the resulting image. Default: PNG.
    background_color
        Background color of the graph. Default: white.
    font_size
        Font size used in the graph. Default: 15.
    layout
        The layout of the graph. Default: Left-to-right.

    Returns
    -------
    dot
        A single Graphviz Graph.
    """
    if not isinstance(process_tree, ProcessTree):
        raise TypeError("the method can be applied only to a 'ProcessTree'!")

    with NamedTemporaryFile(suffix=".gv") as file:
        dot = Graph(
            "pt",
            filename=file.name,
            engine="dot",
            graph_attr={"bgcolor": background_color, "rankdir": layout},
            format=image_format,
        )
        dot.attr("node", shape="ellipse", fixedsize="false")

        # since the process tree object needs to be sorted in the visualization, make a deepcopy of it before proceeding
        tree = deepcopy(process_tree)
        generic.tree_sort(tree)

        visualize_nodes(tree, dot, str(font_size))

        dot.attr(overlap="false")
        dot.attr(splines="false")

        return dot


def visualize_nodes(
    tree: ProcessTree,
    dot: Graph,
    font_size: str,
    edge_label: Optional[str] = None,
) -> None:
    """
    Recursively visualize nodes of a process tree.

    Parameters
    ----------
    tree
        Current Process Tree
    dot
        Graphviz Graph object.
    font_size
        Font size used in the graph.
    edge_label
        If set adds a label to next node.

    Returns
    -------
    dot
        A Graphviz Graph.
    """
    this_node_id = str(id(tree))

    if tree.operator is None:
        if tree.label is None:
            dot.node(
                this_node_id,
                "tau",
                style="filled",
                fillcolor="black",
                shape="point",
                width="0.075",
                fontsize=font_size,
            )
        else:
            dot.node(
                this_node_id,
                str(tree),
                tooltip=str(tree),
                color="black",
                fontcolor="black",
                fontsize=font_size,
            )
    else:
        dot.node(
            this_node_id,
            str(tree.operator),
            tooltip=str(tree.operator),
            color="black",
            fontcolor="black",
            fontsize=font_size,
        )

        for idx, child in enumerate(tree.children):
            label = None
            if tree.operator == Operator.PARTICIPANT:
                label = child.label
            if tree.operator in (Operator.SENT, Operator.RECEIVED):
                if idx > 0:
                    label = child.properties["participant"]
            visualize_nodes(child, dot, font_size, edge_label=label)

    if tree.parent is not None:
        label = "" if edge_label is None else edge_label
        # Shows participant name on the arc
        dot.edge(
            str(id(tree.parent)),
            this_node_id,
            dirType="none",
            label=label,
            fontsize=font_size,
        )
