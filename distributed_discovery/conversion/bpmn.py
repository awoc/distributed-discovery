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

from pm4py.objects.conversion.process_tree.variants.to_bpmn import (
    delete_tau_transitions,
    Counts,
    add_xor_gateway,
    add_tau_task,
    add_task,
    add_parallel_gateway,
    add_inclusive_gateway,
)
from pm4py.objects.bpmn.obj import BPMN

from distributed_discovery.objects.process_tree import ProcessTree
from distributed_discovery.objects.process_tree_operator import Operator
from distributed_discovery.objects.message_flow import MessageFlow


def process_tree_to_bpmn(
    tree: ProcessTree,
) -> Tuple[List[BPMN], Dict[MessageFlow, List[BPMN.BPMNNode]]]:
    """
    Converts a process tree to a list of BPMN graphs for each participant.

    Parameters
    ----------
    tree
        The process tree.

    Returns
    -------
    bpmns
        A list of converted BPMN graphs.
    message_bpmn
        Dictionary mapping activities to BPMN Events.
    """
    if not isinstance(tree, ProcessTree):
        raise TypeError("the method can be applied only to a 'ProcessTree!")

    message_bpmn: Dict[MessageFlow, List[BPMN.Event]] = {}
    bpmns: List[BPMN] = []

    if tree.operator == Operator.PARTICIPANT:
        for sub_graph in tree.children:
            name = sub_graph.label
            bpmn = subgraph_to_bpmn(sub_graph, message_bpmn, name)
            bpmns.append(bpmn)
    else:
        bpmn = subgraph_to_bpmn(tree, message_bpmn, "")
        bpmns.append(bpmn)

    return bpmns, message_bpmn


# flake8: noqa: C901
def recursively_add_tree(
    tree: ProcessTree,
    bpmn: BPMN,
    initial_event: BPMN.Event,
    final_event: BPMN.Event,
    message_bpmn: Dict[MessageFlow, List[BPMN.Event]],
    name: str,
    counts: Counts,
    rec_depth: int,
) -> Tuple[BPMN, Counts, Dict[MessageFlow, List[BPMN.Event]], BPMN.Event, BPMN.Event]:
    """
    Recursively constructs a BPMN graph from a process tree.

    Parameters
    ----------
    tree
        The process tree.
    bpmn
        The BPMN graph that is constructed.
    initial_event
        Event that came before.
    final_event
        Event that follows.
    message_bpmn
        A dictionary for all message events to its BPMN event.
    name
        The participant name.
    counts
        Object to store the number of gateways.
    rec_depth
        Recursion depth.

    Returns
    -------
    bpmn, counts, message_bpmn, initial_connector, final_connector
        The converted BPMN graph, counts object, a dictionary of all message events, the current start and end event.
    """
    tree_children = list(tree.children)
    initial_connector = None
    final_connector = None

    if tree.operator is None:
        trans = tree
        if trans.label is None:
            bpmn, task, counts = add_tau_task(bpmn, counts)
            bpmn.add_flow(BPMN.Flow(initial_event, task))
            bpmn.add_flow(BPMN.Flow(task, final_event))
            initial_connector = task
            final_connector = task
        else:
            bpmn, task, counts = add_task(bpmn, counts, trans.label)
            bpmn.add_flow(BPMN.Flow(initial_event, task))
            bpmn.add_flow(BPMN.Flow(task, final_event))
            initial_connector = task
            final_connector = task

    elif tree.operator == Operator.SENT:
        # First child sender
        message_activity = tree_children[0]
        # Note: First child is sender, therefore > 2
        recipients = len(tree_children) - 1
        # Checks if message was sent to more than one activity
        if recipients > 1:
            # Add Parallel/AND gateway
            bpmn, split_gateway, join_gateway, counts = add_parallel_gateway(
                bpmn, counts
            )
            bpmn.add_flow(BPMN.Flow(initial_event, split_gateway))
            bpmn.add_flow(BPMN.Flow(join_gateway, final_event))

            counts.inc_para_gateways()

            for _ in range(recipients):
                bpmn, task = add_message_intermediate_throw_event(
                    bpmn, message_activity.label, message_bpmn, name
                )
                bpmn.add_flow(BPMN.Flow(split_gateway, task))
                bpmn.add_flow(BPMN.Flow(task, join_gateway))

            initial_connector = join_gateway
            final_connector = join_gateway
        else:
            if isinstance(final_event, BPMN.EndEvent):
                bpmn, task = add_message_end_event(
                    bpmn, message_activity.label, message_bpmn, name
                )
                bpmn.remove_node(final_event)
                bpmn.add_flow(BPMN.Flow(initial_event, task))
            else:
                bpmn, task = add_message_intermediate_throw_event(
                    bpmn, message_activity.label, message_bpmn, name
                )
                bpmn.add_flow(BPMN.Flow(initial_event, task))
                bpmn.add_flow(BPMN.Flow(task, final_event))

            initial_connector = task
            final_connector = task

    elif tree.operator == Operator.RECEIVED:
        message_activity = tree_children[0]

        sender = len(tree_children) - 1
        if sender > 1:
            # Add Parallel/AND gateway
            bpmn, split_gateway, join_gateway, counts = add_parallel_gateway(
                bpmn, counts
            )

            bpmn.add_flow(BPMN.Flow(initial_event, split_gateway))
            bpmn.add_flow(BPMN.Flow(join_gateway, final_event))

            counts.inc_para_gateways()

            for _ in range(sender):
                bpmn, task = add_message_intermediate_catch_event(
                    bpmn, message_activity.label, message_bpmn, name
                )
                bpmn.add_flow(BPMN.Flow(split_gateway, task))
                bpmn.add_flow(BPMN.Flow(task, join_gateway))

            initial_connector = join_gateway
            final_connector = join_gateway
        else:
            if isinstance(initial_event, BPMN.StartEvent):
                bpmn, task = add_message_start_event(
                    bpmn, message_activity.label, message_bpmn, name
                )
                bpmn.remove_node(initial_event)
                bpmn.add_flow(BPMN.Flow(task, final_event))

            else:
                bpmn, task = add_message_intermediate_catch_event(
                    bpmn, message_activity.label, message_bpmn, name
                )

                bpmn.add_flow(BPMN.Flow(initial_event, task))
                bpmn.add_flow(BPMN.Flow(task, final_event))

            initial_connector = task
            final_connector = task

    elif tree.operator in (Operator.XOR, Operator.PARALLEL, Operator.OR):
        if tree.operator == Operator.XOR:
            bpmn, split_gateway, join_gateway, counts = add_xor_gateway(bpmn, counts)
        elif tree.operator == Operator.PARALLEL:
            bpmn, split_gateway, join_gateway, counts = add_parallel_gateway(
                bpmn, counts
            )
        else:
            bpmn, split_gateway, join_gateway, counts = add_inclusive_gateway(
                bpmn, counts
            )

        for subtree in tree_children:
            bpmn, counts, message_bpmn, _, _ = recursively_add_tree(
                subtree,
                bpmn,
                split_gateway,
                join_gateway,
                message_bpmn,
                name,
                counts,
                rec_depth + 1,
            )
        bpmn.add_flow(BPMN.Flow(initial_event, split_gateway))
        bpmn.add_flow(BPMN.Flow(join_gateway, final_event))
        initial_connector = split_gateway
        final_connector = join_gateway

    elif tree.operator == Operator.SEQUENCE:
        initial_intermediate_task = initial_event
        bpmn, final_intermediate_task, counts = add_tau_task(bpmn, counts)
        for i, child in enumerate(tree_children):
            (
                bpmn,
                counts,
                message_bpmn,
                initial_connect,
                final_connect,
            ) = recursively_add_tree(
                child,
                bpmn,
                initial_intermediate_task,
                final_intermediate_task,
                message_bpmn,
                name,
                counts,
                rec_depth + 1,
            )
            initial_intermediate_task = final_connect
            if i == 0:
                initial_connector = initial_connect
            if i == len(tree_children) - 2:
                final_intermediate_task = final_event
            else:
                bpmn, final_intermediate_task, counts = add_tau_task(bpmn, counts)
            final_connector = final_connect

    elif tree.operator == Operator.LOOP:
        if len(tree_children) != 2:
            raise Exception("Loop doesn't have 2 children")

        do = tree_children[0]
        redo = tree_children[1]
        bpmn, split, join, counts = add_xor_gateway(bpmn, counts)
        bpmn, counts, message_bpmn, _, _ = recursively_add_tree(
            do, bpmn, join, split, message_bpmn, name, counts, rec_depth + 1
        )
        bpmn, counts, message_bpmn, _, _ = recursively_add_tree(
            redo, bpmn, split, join, message_bpmn, name, counts, rec_depth + 1
        )
        bpmn.add_flow(BPMN.Flow(initial_event, join))
        bpmn.add_flow(BPMN.Flow(split, final_event))
        initial_connector = join
        final_connector = split

    return bpmn, counts, message_bpmn, initial_connector, final_connector


def subgraph_to_bpmn(
    tree: ProcessTree, message_bpmn: Dict[MessageFlow, List[BPMN.Event]], name: str
) -> BPMN:
    """
    Converts a process tree subgraph of a participant to a BPMN graph and adds in-place the event to a dictionary.

    Parameters
    ----------
    tree
        The process tree subgraph.
    message_bpmn
    name
        The participant name of the subgraph.

    Returns
    -------
    bpmn
        The converted BPMN graph.
    """
    counts = Counts()
    bpmn = BPMN(name=name)
    start_event = BPMN.StartEvent(name="start", isInterrupting=True)
    end_event = BPMN.NormalEndEvent(name="end")
    bpmn.add_node(start_event)
    bpmn.add_node(end_event)
    bpmn, counts, _, _, _ = recursively_add_tree(
        tree, bpmn, start_event, end_event, message_bpmn, name, counts, 0
    )
    bpmn = delete_tau_transitions(bpmn, counts)

    for node in bpmn.get_nodes():
        node.set_process(bpmn.get_process_id())

    for edge in bpmn.get_flows():
        edge.set_process(bpmn.get_process_id())

    return bpmn


def add_message_intermediate_throw_event(
    bpmn: BPMN, label: str, message_bpmn: Dict[MessageFlow, List[BPMN.Event]], name: str
) -> Tuple[BPMN, BPMN.MessageIntermediateThrowEvent]:
    """
    Creates and adds a message intermediate throw event to a BPMN graph.

    Parameters
    ----------
    bpmn
        The BPMN graph.
    label
        The name of the message event.
    message_bpmn
        Dictionary of Event name to BPMN Node.
    name
         The participant name

    Returns
    -------
    bpmn, event
        Updated BPMN graph and the created message intermediate throw event.

    """
    task = BPMN.MessageIntermediateThrowEvent(name=label)
    bpmn.add_node(task)

    add_message_event(task, label, message_bpmn, name)

    return bpmn, task


def add_message_end_event(
    bpmn: BPMN, label: str, message_bpmn: Dict[MessageFlow, List[BPMN.Event]], name: str
) -> Tuple[BPMN, BPMN.MessageEndEvent]:
    """
    Creates and adds a message end event to a BPMN graph.

    Parameters
    ----------
    bpmn
        The BPMN graph.
    label
        The name of the message event.
    message_bpmn
        Dictionary of Event name to BPMN Node.
    name
         The participant name

    Returns
    -------
    bpmn, event
        Updated BPMN graph and the created message end event.

    """
    end_event = BPMN.MessageEndEvent(name=label)
    bpmn.add_node(end_event)

    add_message_event(end_event, label, message_bpmn, name)

    return bpmn, end_event


def add_message_start_event(
    bpmn: BPMN, label: str, message_bpmn: Dict[MessageFlow, List[BPMN.Event]], name: str
) -> Tuple[BPMN, BPMN.MessageStartEvent]:
    """
    Creates and adds a message start event to a BPMN graph.

    Parameters
    ----------
    bpmn
        The BPMN graph.
    label
        The name of the message event.
    message_bpmn
        Dictionary of Event name to BPMN Node.
    name
         The participant name

    Returns
    -------
    bpmn, event
        Updated BPMN graph and the created message start event.

    """
    start_event = BPMN.MessageStartEvent(name=label)
    bpmn.add_node(start_event)

    add_message_event(start_event, label, message_bpmn, name)

    return bpmn, start_event


def add_message_intermediate_catch_event(
    bpmn: BPMN, label: str, message_bpmn: Dict[MessageFlow, List[BPMN.Event]], name: str
) -> Tuple[BPMN, BPMN.MessageIntermediateCatchEvent]:
    """
    Creates and adds an intermediate catch event to a BPMN graph.

    Parameters
    ----------
    bpmn
        The BPMN graph.
    label
        The name of the message event.
    message_bpmn
        Dictionary of Event name to BPMN Node.
    name
         The participant name

    Returns
    -------
    bpmn, event
        Updated BPMN graph and the created intermediate catch event.

    """
    event = BPMN.MessageIntermediateCatchEvent(name=label)
    bpmn.add_node(event)

    add_message_event(event, label, message_bpmn, name)

    return bpmn, event


def add_message_event(
    event: BPMN.Event,
    label: str,
    message_bpmn: Dict[MessageFlow, List[BPMN.Event]],
    name: str,
) -> None:
    """
    Adds in-place a message event to a dictionary of all message events.

    Parameters
    ----------
    event
        The message event.
    label
        The name of the event.
    message_bpmn
        Dictionary of Event name to BPMN Node.
    name
        The participant name.

    Returns
    -------
    """
    message_flow = MessageFlow(name, label)

    if message_flow in message_bpmn:
        prev_tasks = message_bpmn[message_flow]
        prev_tasks.append(event)
    else:
        message_bpmn[message_flow] = [event]
