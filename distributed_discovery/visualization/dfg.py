"""
    This file contains modified code from PM4Py
    to visualize DFGs for distributed processes.

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
from typing import Dict, Tuple, Optional, Set
from collections import Counter
from copy import copy

from graphviz import Digraph
from pm4py.objects.dfg.utils import dfg_utils
from pm4py.visualization.dfg.variants.frequency import (
    get_activities_color,
    assign_penwidth_edges,
)
from pm4py.visualization.common.utils import human_readable_stat
from pm4py.objects.log.obj import EventLog
from pm4py.statistics.start_activities.log.get import get_start_activities
from pm4py.statistics.end_activities.log.get import get_end_activities

from distributed_discovery.objects.dfg import DistributedDfg, DfgType
from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.util.filter import get_participants_event_logs


def visualize_dfg(
    dfg: DistributedDfg,
    image_format: str = "png",
    background_color: str = "white",
    font_size: int = 12,
    layout: str = "LR",
) -> Digraph:
    """
    Visualizes a directly follows graph for a distributed event log as a Graphviz Digraph
    containing clusters for each participant in the log.

    Parameters
    ----------
    dfg
        A directly follows graph for each participant in a distributed event log.
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
        A Graphviz Digraph containing a subgraph for each participant in the DFG.
    """
    if not isinstance(dfg, DistributedDfg):
        raise TypeError("the method can be applied only to a 'DistributedDfg'!")

    font_size = str(font_size)

    participant_logs = get_participants_event_logs(dfg.event_log)

    with NamedTemporaryFile(suffix=".gv") as file:
        parent_viz = Digraph(
            "parent",
            filename=file.name,
            engine="dot",
            graph_attr={"rankdir": layout, "bgcolor": background_color},
            format=image_format,
        )

        (
            participants_activities_count,
            participants_activities_delivered_messages_count,
        ) = generate_message_activities_count(
            dfg.messages_dfg, dfg.event_log.participants
        )

        for participant in dfg.event_log.participants:
            participant_log = participant_logs[participant]
            # send_activities_delivered_messages = dfg.messages_dfg[participant]

            cluster_viz = dfg_to_graphviz(
                participant_log,
                participant,
                dfg.participant_dfgs[participant],
                dfg.type,
                participants_activities_count[participant],
                participants_activities_delivered_messages_count[participant],
                background_color,
                font_size,
            )

            parent_viz.subgraph(cluster_viz)

            del cluster_viz

        # Add message flow edges
        pen_width = assign_penwidth_edges(dfg.messages_dfg)

        for message_exchange, value in dfg.messages_dfg.items():
            sender, receiver = message_exchange
            if sender is not None and receiver is not None:
                label = (
                    str(value)
                    if dfg.type == DfgType.FREQUENCY
                    else human_readable_stat(value, {})
                )

                parent_viz.edge(
                    str(hash(f"{sender.participant}_{sender.activity}")),
                    str(hash(f"{receiver.participant}_{receiver.activity}")),
                    label=label,
                    font_size=font_size,
                    penwidth=str(pen_width[message_exchange]),
                    _attributes={"style": "dotted"},
                    fontsize=font_size,
                )

        parent_viz.attr(overlap="false")

        return parent_viz


def generate_message_activities_count(
    messages: Dict[Tuple[MessageFlow, MessageFlow], int], participants: [str]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    Counts the number sent and received messages per activity
    and also the number of successfully delivered messages per participant.

    Parameters
    ----------
    messages
        The count of message flows.
    participants
        List of participants.

    Returns
    -------
    participants_activities_count
        Count of attributes in the log per participant.
    participants_activities_delivered_messages_count
        Count of successfully delivered messages for each send activity per participant.
    """
    participants_activities_count = {participant: {} for participant in participants}

    participants_activities_delivered_messages_count = {
        participant: {} for participant in participants
    }

    for sender, receiver in messages.keys():
        if sender is not None:
            send_activities_count = participants_activities_count[sender.participant]

            if sender.activity in send_activities_count:
                send_activities_count[sender.activity] += messages[(sender, receiver)]
            else:
                send_activities_count[sender.activity] = messages[(sender, receiver)]

        if receiver is not None:
            receive_activities_count = participants_activities_count[
                receiver.participant
            ]

            if receiver.activity in receive_activities_count:
                receive_activities_count[receiver.activity] += messages[
                    (sender, receiver)
                ]
            else:
                receive_activities_count[receiver.activity] = messages[
                    (sender, receiver)
                ]

        if sender is not None and receiver is not None:
            delivered_messages_count = participants_activities_delivered_messages_count[
                sender.participant
            ]

            if sender.activity in delivered_messages_count:
                delivered_messages_count[sender.activity] += messages[
                    (sender, receiver)
                ]
            else:
                delivered_messages_count[sender.activity] = messages[(sender, receiver)]

    return (
        participants_activities_count,
        participants_activities_delivered_messages_count,
    )


def dfg_to_graphviz(
    log: EventLog,
    participant: str,
    dfg: Dict[Tuple[str, str], int],
    measure: DfgType,
    message_activities_count: Dict[str, int],
    delivered_message_activities_count: Dict[str, int],
    background_color: str,
    font_size: str,
) -> Digraph:
    """
    Visualizes the subgraph of a participant in a directly-follows-graph.

    Parameters
    ----------
    log
        Event log used to find start and end activities.
    participant
        The name of the participant of the log.
    dfg
        Frequency Directly-follows graph.
    measure
        The type of the DFG.
    message_activities_count
        Count of messages received in the log.
    delivered_message_activities_count
        Count of successfully delivered messages for each send activity.
    background_color
        Background color of the graph used by Graphviz.
    font_size
        Font size of elements in the graph.

    Returns
    -------
    viz
        Digraph cluster.
    """
    max_no_of_edges_in_diagram = 100000
    activities = dfg_utils.get_activities_from_dfg(dfg)

    start_activities = get_start_activities(log)
    end_activities = get_end_activities(log)

    # the frequency of an activity in the log is at least the number of occurrences of
    # incoming arcs in the DFG.
    # if the frequency of the start activities nodes is also provided, use also that.
    activities_count = Counter({key: 0 for key in activities})
    for el in dfg:
        activities_count[el[1]] += dfg[el]

    if isinstance(start_activities, dict):
        for act in start_activities:
            activities_count[act] += start_activities[act]

    return graphviz_visualization(
        activities_count,
        message_activities_count,
        delivered_message_activities_count,
        dfg,
        measure,
        participant,
        max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
        start_activities=start_activities,
        end_activities=end_activities,
        background_color=background_color,
        font_size=font_size,
    )


def visualize_start_end_activities(
    dot: Digraph,
    start_activities: Dict[str, int],
    end_activities: Dict[str, int],
    activities_map: Dict[str, str],
    participant: str,
    font_size: str,
) -> None:
    """
    Adds start and end activities to a DFG graph.

    Parameters
    ----------
    dot
        Digraph object.
    start_activities
        Start activities of the event log.
    end_activities
        End activities of the event log.
    activities_map
        Dictionary of activity names with their name in the Digraph.
    participant
        Name of the participant.
    font_size
        Font size of elements in the graph.
    Returns
    -------

    """
    start_activities_to_include = [
        act for act in start_activities if act in activities_map
    ]
    end_activities_to_include = [act for act in end_activities if act in activities_map]

    if start_activities_to_include:
        dot.node(
            f"{participant}_@@startnode",
            "<&#9679;>",
            tooltip="Start Activity",
            shape="circle",
            fontsize="34",
        )
        for act in start_activities_to_include:
            label = (
                str(start_activities[act]) if isinstance(start_activities, dict) else ""
            )
            dot.edge(
                f"{participant}_@@startnode",
                activities_map[act],
                label=label,
                fontsize=font_size,
            )

    if end_activities_to_include:
        dot.node(
            f"{participant}_@@endnode",
            "<&#9632;>",
            tooltip="End Activity",
            shape="doublecircle",
            fontsize="32",
        )
        for act in end_activities_to_include:
            label = str(end_activities[act]) if isinstance(end_activities, dict) else ""
            dot.edge(
                activities_map[act],
                f"{participant}_@@endnode",
                label=label,
                fontsize=font_size,
            )


def graphviz_visualization(
    activities_count: Dict[str, int],
    message_activities_count: Dict[str, int],
    delivered_message_activities_count: Dict[str, int],
    dfg: Dict[Tuple[str, str], int],
    measure: DfgType,
    participant: str,
    max_no_of_edges_in_diagram,
    background_color: str,
    font_size: str,
    start_activities: Optional[Dict[str, int]] = None,
    end_activities: Optional[Dict[str, int]] = None,
) -> Digraph:
    """
    Visualizes a DFG of a participant.

    Parameters
    ----------
    activities_count
        Count of attributes in the log.
    message_activities_count
        Count of messages received in the log.
    delivered_message_activities_count
        Count of successfully delivered messages for each send activity.
    dfg
        The DFG graph.
    measure
        The type of the DFG.
    participant
        Name of the participant.
    max_no_of_edges_in_diagram
        Maximum number of edges in the diagram allowed for visualization.
    background_color
        Background color of the graph.
    font_size
        Font size of elements in the graph.
    start_activities
        Start activities of the log.
    end_activities
        End activities of the log.

    Returns
    -------
    dot
        Digraph object.
    """
    if start_activities is None:
        start_activities = {}
    if end_activities is None:
        end_activities = {}

    with NamedTemporaryFile(suffix=".gv") as file:
        dot = Digraph(
            f"cluster_{participant}",
            filename=file.name,
            engine="dot",
            graph_attr={
                "label": participant,
                "tooltip": participant,
                "bgcolor": background_color,
                "fontsize": font_size,
            },
        )

        # first, remove edges in diagram that exceeds the maximum number of edges in the diagram
        dfg_key_value_list = []
        for edge in dfg:
            dfg_key_value_list.append([edge, dfg[edge]])
        # more fine-grained sorting to avoid that edges that are below the threshold are
        # non-deterministically removed
        dfg_key_value_list = sorted(
            dfg_key_value_list, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True
        )
        dfg_key_value_list = dfg_key_value_list[
            0 : min(len(dfg_key_value_list), max_no_of_edges_in_diagram)
        ]
        dfg_allowed_keys = [x[0] for x in dfg_key_value_list]
        dfg_keys = list(dfg.keys())

        for edge in dfg_keys:
            if edge not in dfg_allowed_keys:
                del dfg[edge]

        # calculate edges penwidth
        penwidth = assign_penwidth_edges(dfg)
        activities_in_dfg = set()
        activities_count_int = copy(activities_count)

        for edge in dfg:
            activities_in_dfg.add(edge[0])
            activities_in_dfg.add(edge[1])

        if measure == DfgType.FREQUENCY:
            # assign attributes color
            activities_color = get_activities_color(activities_count_int)
            get_delivered_activities_color(
                activities_color,
                delivered_message_activities_count,
                message_activities_count,
            )

            activities_map = visualize_frequency_nodes(
                dot,
                activities_in_dfg,
                participant,
                message_activities_count,
                activities_count_int,
                activities_color,
                font_size,
            )
        else:
            activities_map = visualize_performance_nodes(
                dot, activities_in_dfg, participant, activities_count_int, font_size
            )

        # represent edges
        visualize_edges(dot, dfg, participant, font_size, penwidth, measure)

        visualize_start_end_activities(
            dot,
            start_activities,
            end_activities,
            activities_map,
            participant,
            font_size,
        )

        return dot


def visualize_frequency_nodes(
    dot: Digraph,
    activities: Set[str],
    participant: str,
    message_activities_count: Dict[str, int],
    activities_count: Dict[str, int],
    color: Dict[str, str],
    font_size: str,
) -> Dict[str, str]:
    """
    Adds nodes to a DFG and annotates them with frequency information.

    Parameters
    ----------
    dot
        The Digraph.
    activities
        Set of activities in the DFG.
    participant
        Name of the participant.
    activities_count
        Count of attributes in the log.
    message_activities_count
        Count of messages received in the log.
    color
        Dictionary of colors for nodes.
    font_size
        Font size of nodes in the graph.

    Returns
    -------
    activities_map
        Dictionary of activity names with their name in the Digraph.
    """
    dot.attr("node", shape="box")

    if len(activities) == 0:
        activities_to_include = sorted(list(set(activities_count)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities)))

    activities_map: Dict[str, str] = {}

    for act in activities_to_include:
        node_name = f"{participant}_{act}"
        hashed_name = str(hash(node_name))

        activity_count = activities_count[act]
        message_activity_count = (
            message_activities_count[act] if act in message_activities_count else 0
        )

        dot.node(
            hashed_name,
            f"{act} ({activity_count}, {message_activity_count})",
            style="filled",
            fillcolor=color[act],
            tooltip=act,
            fontsize=font_size,
        )
        activities_map[act] = hashed_name

    return activities_map


def visualize_performance_nodes(
    dot: Digraph,
    activities: Set[str],
    participant: str,
    activities_count: Dict[str, int],
    font_size: str,
) -> Dict[str, str]:
    """
    Adds nodes to a DFG without any additional annotations.

    Parameters
    ----------
    dot
         The Digraph.
    activities
        Set of activities in the DFG.
    participant
        Name of the participant.
    activities_count
        Count of attributes in the log.
    font_size
        Font size of nodes in the graph.

    Returns
    -------
    activities_map
        Dictionary of activity names with their name in the Digraph.
    """
    dot.attr("node", shape="box")

    if len(activities) == 0:
        activities_to_include = sorted(list(set(activities_count)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities)))

    activities_map = {}

    for act in activities_to_include:
        node_name = f"{participant}_{act}"
        hashed_name = str(hash(node_name))

        dot.node(
            hashed_name,
            f"{act}",
            tooltip=act,
            fontsize=font_size,
        )
        activities_map[act] = hashed_name

    return activities_map


def visualize_edges(
    dot: Digraph,
    dfg: Dict[Tuple[str, str], int],
    participant: str,
    font_size: str,
    penwidth: Dict[Tuple[str, str], str],
    measure: DfgType,
) -> None:
    """
    Visualizes edges of a DFG.

    Parameters
    ----------
    dot
        The Digraph.
    dfg
    participant
        Name of the participant.
    font_size
        Font size of nodes in the graph.
    penwidth
        Dictionary of edges and their width.
    measure
        The type of the DFG.

    Returns
    -------

    """
    # make edges addition always in the same order
    edges = sorted(list(dfg.keys()))

    for edge in edges:
        label = (
            str(dfg[edge])
            if measure == DfgType.FREQUENCY
            else human_readable_stat(dfg[edge], {})
        )

        dot.edge(
            str(hash(f"{participant}_{edge[0]}")),
            str(hash(f"{participant}_{edge[1]}")),
            label=label,
            penwidth=str(penwidth[edge]),
            fontsize=font_size,
        )


def get_delivered_activities_color(
    activities_color: Dict[str, str],
    delivered_message_activities_count: Dict[str, int],
    message_activities_count: Dict[str, int],
) -> None:
    """
    Calculates the color for sending activities if messages are dropped.

    Parameters
    ----------
    activities_color
        Existing dictionary for activity colors.
    delivered_message_activities_count
        Count of successfully delivered messages for each send activity.
    message_activities_count
        Count of messages received in the log.

    Returns
    -------

    """
    for activity, count in delivered_message_activities_count.items():
        dropped_message_rate = count / message_activities_count[activity]

        if dropped_message_rate < 1:
            # Calculate transparency
            activities_color[
                activity
            ] = f"#FF0000{int((1 - dropped_message_rate) * 255):X}"
