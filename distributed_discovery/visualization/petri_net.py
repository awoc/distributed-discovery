"""
    This file contains modified code from PM4Py
    to visualize workflow modules.

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
from uuid import uuid4
from typing import List, Tuple, Dict, Set

from graphviz import Digraph
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.obj import PetriNet, Marking

from distributed_discovery.objects.message_flow import MessageFlow


def visualize_petri_net(
    petri_nets: List[Tuple[str, PetriNet, Marking, Marking]],
    message_petri_net_per_participant: Dict[MessageFlow, PetriNet.Transition],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
    image_format: str = "png",
    background_color: str = "white",
    font_size: int = 12,
    layout: str = "LR",
) -> Digraph:
    """
    Visualizes petri nets for a distributed event log as a Graphviz Digraph.

    Parameters
    ----------
    petri_nets
        List of petri nets of each participant with start and end markings.
    message_petri_net_per_participant
        A dictionary of message flows to a petri net transition.
    sent_messages
        Sent messages per participant.
    image_format
        Format of the resulting image. Default: PNG.
    background_color
        Background color of the graph. Default: white.
    font_size
        Font size used in the graph. Default: 12.
    layout
        The layout of the graph. Default: Left-to-right.

    Returns
    -------
    parent_viz
        A Graphviz Digraph containing a subgraph for each participant in the petri net.
    """
    font_size = str(font_size)

    with NamedTemporaryFile(suffix=".gv") as file:

        parent_graph = Digraph(
            name="parent",
            filename=file.name,
            graph_attr={"bgcolor": background_color, "rankdir": layout},
            format=image_format,
        )

        for name, petri_net, initial_marking, final_marking in petri_nets:
            cluster_viz = petri_net_to_graphviz(
                petri_net,
                initial_marking,
                final_marking,
                layout,
                font_size,
                background_color,
            )
            parent_graph.subgraph(cluster_viz)

            del cluster_viz

        for participant, entries in sent_messages.items():
            for sender, receivers in entries.items():
                for receiver in receivers:
                    place_id = str(uuid4())
                    parent_graph.node(
                        place_id, "", shape="circle", fixedsize="true", width="0.75"
                    )

                    sender_transition = message_petri_net_per_participant[
                        MessageFlow(participant, sender)
                    ]
                    receiver_transition = message_petri_net_per_participant[receiver]

                    # Sender
                    parent_graph.edge(
                        str(id(sender_transition)),
                        place_id,
                        fontsize=font_size,
                        arrowhead="normal",
                    )
                    # Receiver
                    parent_graph.edge(
                        place_id,
                        str(id(receiver_transition)),
                        fontsize=font_size,
                        arrowhead="normal",
                    )

        parent_graph.attr(overlap="false")

        return parent_graph


def visualize_transitions(dot: Digraph, net: PetriNet, font_size: str) -> None:
    """
    Visualizes transition in a petri net.

    Parameters
    ----------
    dot
        The Digraph.
    net
        The petri net.
    font_size
        Font size used in the graph.

    Returns
    -------

    """
    dot.attr("node", shape="box")
    for t in net.transitions:
        if t.label is not None:
            dot.node(str(id(t)), str(t.label), fontsize=font_size)
        else:
            dot.node(
                str(id(t)),
                label="",
                style="filled",
                fillcolor="black",
                fontsize=font_size,
            )

        if petri_properties.TRANS_GUARD in t.properties:
            guard = t.properties[petri_properties.TRANS_GUARD]
            dot.node(str(id(t)) + "guard", style="dotted", label=guard)
            dot.edge(str(id(t)) + "guard", str(id(t)), arrowhead="none", style="dotted")


def visualize_places(
    dot: Digraph, net: PetriNet, initial_marking: Marking, final_marking: Marking
) -> None:
    """
    Visualizes places in a petri net.

    Parameters
    ----------
    dot
        The Digraph.
    net
        The petri net.
    initial_marking
        The initial marking in the petri net.
    final_marking
        The final marking in the petri net.

    Returns
    -------

    """
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted(
        [x for x in list(net.places) if x in initial_marking], key=lambda x: x.name
    )
    places_sort_list_fm = sorted(
        [
            x
            for x in list(net.places)
            if x in final_marking and x not in initial_marking
        ],
        key=lambda x: x.name,
    )
    places_sort_list_not_im_fm = sorted(
        [
            x
            for x in list(net.places)
            if x not in initial_marking and x not in final_marking
        ],
        key=lambda x: x.name,
    )
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places belonging neither to the initial marking nor to the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = (
        places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm
    )

    for p in places_sort_list:
        if p in initial_marking:
            if initial_marking[p] == 1:
                dot.node(
                    str(id(p)),
                    "<&#9679;>",
                    tooltip="Initial Marking",
                    fontsize="34",
                    fixedsize="true",
                    shape="circle",
                    width="0.75",
                )
            else:
                dot.node(
                    str(id(p)),
                    str(initial_marking[p]),
                    fontsize="34",
                    fixedsize="true",
                    shape="circle",
                    width="0.75",
                )
        elif p in final_marking:
            # <&#9632;>
            dot.node(
                str(id(p)),
                "<&#9632;>",
                tooltip="Final Marking",
                fontsize="32",
                shape="doublecircle",
                fixedsize="true",
                width="0.75",
            )
        else:
            dot.node(str(id(p)), "", shape="circle", fixedsize="true", width="0.75")


def visualize_arcs(dot: Digraph, net: PetriNet, font_size: str) -> None:
    """
    Visualizes arcs in a petri net.

    Parameters
    ----------
    dot
        The Digraph.
    net
        The petri net.
    font_size
        Font size used in the graph.

    Returns
    -------

    """
    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(
        list(net.arcs), key=lambda x: (x.source.name, x.target.name)
    )

    # check if there is an arc with weight different than 1.
    # in that case, all the arcs in the visualization should have the arc weight visible
    arc_weight_visible = False
    for arc in arcs_sort_list:
        if arc.weight != 1:
            arc_weight_visible = True
            break

    for a in arcs_sort_list:
        arrowhead = "normal"
        if petri_properties.ARCTYPE in a.properties:
            if a.properties[petri_properties.ARCTYPE] == petri_properties.RESET_ARC:
                arrowhead = "vee"
            elif (
                a.properties[petri_properties.ARCTYPE] == petri_properties.INHIBITOR_ARC
            ):
                arrowhead = "dot"
        else:
            if arc_weight_visible:
                dot.edge(
                    str(id(a.source)),
                    str(id(a.target)),
                    fontsize=font_size,
                    arrowhead=arrowhead,
                    label=str(a.weight),
                )
            else:
                dot.edge(
                    str(id(a.source)),
                    str(id(a.target)),
                    fontsize=font_size,
                    arrowhead=arrowhead,
                )


def petri_net_to_graphviz(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    layout: str,
    font_size: str,
    background_color: str,
) -> Digraph:
    """
    Visualizes a Petri Net as a cluster.

    Parameters
    ----------
    net
        Petri Net.
    initial_marking
        Initial Place in the graph.
    final_marking
        Final Place in the graph.
    layout
        The layout of the graph.
    font_size
        Font size used in the graph.
    background_color
        Background color of the graph.

    Returns
    -------
    dot
        A Graphviz Digraph object.

    """
    with NamedTemporaryFile(suffix=".gv") as file:
        dot = Digraph(
            f"cluster_{net.name}",
            filename=file.name,
            engine="dot",
            graph_attr={
                "bgcolor": background_color,
                "label": net.name,
                "tooltip": net.name,
                "rankdir": layout,
                "fontsize": font_size,
            },
        )

        visualize_transitions(dot, net, font_size)
        visualize_places(dot, net, initial_marking, final_marking)
        visualize_arcs(dot, net, font_size)

        dot.attr(overlap="false")

        return dot
