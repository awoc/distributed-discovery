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
from typing import List, Dict, Tuple, Optional

from pm4py.objects.conversion.process_tree.variants.to_petri_net import (
    get_new_place,
    get_new_hidden_trans,
    get_transition,
    check_tau_mandatory_at_final_marking,
    check_tau_mandatory_at_initial_marking,
    Counts,
)
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, remove_place
from pm4py.objects.petri_net.utils import reduction

from distributed_discovery.objects.process_tree import ProcessTree
from distributed_discovery.objects.process_tree_operator import Operator
from distributed_discovery.objects.message_flow import MessageFlow


def process_tree_to_petri_net(
    process_tree: ProcessTree,
) -> Tuple[
    List[Tuple[str, PetriNet, Marking, Marking]], Dict[MessageFlow, PetriNet.Transition]
]:
    """
    Converts a petri net including participants and message flows to multiple petri nets per participant.

    Parameters
    ----------
    process_tree
        The process tree to convert.

    Returns
    -------
    petri_nets
        A list of petri nets per participant.
    message_petri_net
         A dictionary of message flows to a petri net transition.
    """
    if not isinstance(process_tree, ProcessTree):
        raise TypeError("the method can be applied only to a 'ProcessTree!")

    petri_nets: List[Tuple[str, PetriNet, Marking, Marking]] = []
    message_petri_net: Dict[MessageFlow, PetriNet.Transition] = {}

    if process_tree.operator == Operator.PARTICIPANT:
        for sub_graph in process_tree.children:
            participant = sub_graph.label
            (
                petri_net,
                message_petri_net,
                initial_marking,
                final_marking,
            ) = subgraph_to_petri_net(sub_graph, message_petri_net, participant)
            petri_nets.append((participant, petri_net, initial_marking, final_marking))
    else:
        petri_net, _, initial_marking, final_marking = subgraph_to_petri_net(
            process_tree
        )
        petri_nets.append(("", petri_net, initial_marking, final_marking))

    return petri_nets, message_petri_net


# flake8: noqa: C901
def recursively_add_tree(
    tree: ProcessTree,
    net: PetriNet,
    message_petri_net: Dict[MessageFlow, PetriNet.Transition],
    initial_entity_subtree: Optional[PetriNet],
    final_entity_subtree: Optional[PetriNet],
    counts: Counts,
    rec_depth: int,
    participant: str,
    force_add_skip: bool = False,
) -> Tuple[PetriNet, Dict[MessageFlow, PetriNet.Transition], Counts, PetriNet.Place]:
    """
    Recursively add the subtrees to the Petri net

    Parameters
    -----------
    tree
        Current subtree
    net
        Petri net
    message_petri_net
        A dictionary of message flows to a petri net transition.
    initial_entity_subtree
        Initial entity (place/transition) that should be attached from the subtree
    final_entity_subtree
        Final entity (place/transition) that should be attached from the subtree
    counts
        Counts object (keeps the number of places, transitions and hidden transitions)
    rec_depth
        Recursion depth of the current iteration
    participant
        The participant of the petri net.
    force_add_skip
        Boolean value that tells if the addition of a skip is mandatory

    Returns
    ----------
    net
        Updated Petri net
    message_petri_net
        A dictionary of message flows to a petri net transition.
    counts
        Updated counts object (keeps the number of places, transitions and hidden transitions)
    final_place
        Last place added in this recursion
    """
    if isinstance(initial_entity_subtree, PetriNet.Transition):
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        add_arc_from_to(initial_entity_subtree, initial_place, net)
    else:
        initial_place = initial_entity_subtree
    if final_entity_subtree is not None and isinstance(
        final_entity_subtree, PetriNet.Place
    ):
        final_place = final_entity_subtree
    else:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        if final_entity_subtree is not None and isinstance(
            final_entity_subtree, PetriNet.Transition
        ):
            add_arc_from_to(final_place, final_entity_subtree, net)
    tree_children = list(tree.children)

    if force_add_skip:
        invisible = get_new_hidden_trans(counts, type_trans="skip")
        add_arc_from_to(initial_place, invisible, net)
        add_arc_from_to(invisible, final_place, net)

    if tree.operator is None:
        trans = tree
        if trans.label is None:
            petri_trans = get_new_hidden_trans(counts, type_trans="skip")
        else:
            petri_trans = get_transition(counts, trans.label)
        net.transitions.add(petri_trans)
        add_arc_from_to(initial_place, petri_trans, net)
        add_arc_from_to(petri_trans, final_place, net)

    elif tree.operator in (Operator.SENT, Operator.RECEIVED):
        # First child -> sender/receiver
        message_activity = tree_children[0]
        name = message_activity.label

        petri_trans = get_transition(counts, name)
        net.transitions.add(petri_trans)
        add_arc_from_to(initial_place, petri_trans, net)
        add_arc_from_to(petri_trans, final_place, net)

        message_petri_net[MessageFlow(participant, name)] = petri_trans

    elif tree.operator == Operator.XOR:
        for subtree in tree_children:
            net, message_petri_net, counts, intermediate_place = recursively_add_tree(
                subtree,
                net,
                message_petri_net,
                initial_place,
                final_place,
                counts,
                rec_depth + 1,
                participant,
            )
    elif tree.operator == Operator.OR:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)
        terminal_place = get_new_place(counts)
        net.places.add(terminal_place)
        add_arc_from_to(terminal_place, new_final_trans, net)
        first_place = get_new_place(counts)
        net.places.add(first_place)
        add_arc_from_to(new_initial_trans, first_place, net)

        for subtree in tree_children:
            subtree_init_place = get_new_place(counts)
            net.places.add(subtree_init_place)
            add_arc_from_to(new_initial_trans, subtree_init_place, net)
            subtree_start_place = get_new_place(counts)
            net.places.add(subtree_start_place)
            subtree_end_place = get_new_place(counts)
            net.places.add(subtree_end_place)
            trans_start = get_new_hidden_trans(counts, type_trans="inclusiveStart")
            trans_later = get_new_hidden_trans(counts, type_trans="inclusiveLater")
            trans_skip = get_new_hidden_trans(counts, type_trans="inclusiveSkip")
            net.transitions.add(trans_start)
            net.transitions.add(trans_later)
            net.transitions.add(trans_skip)
            add_arc_from_to(first_place, trans_start, net)
            add_arc_from_to(subtree_init_place, trans_start, net)
            add_arc_from_to(trans_start, subtree_start_place, net)
            add_arc_from_to(trans_start, terminal_place, net)

            add_arc_from_to(terminal_place, trans_later, net)
            add_arc_from_to(subtree_init_place, trans_later, net)
            add_arc_from_to(trans_later, subtree_start_place, net)
            add_arc_from_to(trans_later, terminal_place, net)

            add_arc_from_to(terminal_place, trans_skip, net)
            add_arc_from_to(subtree_init_place, trans_skip, net)
            add_arc_from_to(trans_skip, terminal_place, net)
            add_arc_from_to(trans_skip, subtree_end_place, net)

            add_arc_from_to(subtree_end_place, new_final_trans, net)

            net, message_petri_net, counts, intermediate_place = recursively_add_tree(
                subtree,
                net,
                message_petri_net,
                subtree_start_place,
                subtree_end_place,
                counts,
                rec_depth + 1,
                participant,
            )

    elif tree.operator == Operator.PARALLEL:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)

        for subtree in tree_children:
            net, message_petri_net, counts, intermediate_place = recursively_add_tree(
                subtree,
                net,
                message_petri_net,
                new_initial_trans,
                new_final_trans,
                counts,
                rec_depth + 1,
                participant,
            )

    elif tree.operator == Operator.INTERLEAVING:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)

        control_place = get_new_place(counts)
        net.places.add(control_place)

        add_arc_from_to(new_initial_trans, control_place, net)
        add_arc_from_to(control_place, new_final_trans, net)

        for subtree in tree_children:
            placeI = get_new_place(counts)
            net.places.add(placeI)
            iTrans = get_new_hidden_trans(counts, type_trans="iTrans")
            net.transitions.add(iTrans)
            placeF = get_new_place(counts)
            net.places.add(placeF)
            fTrans = get_new_hidden_trans(counts, type_trans="fTrans")
            net.transitions.add(fTrans)

            add_arc_from_to(new_initial_trans, placeI, net)
            add_arc_from_to(placeI, iTrans, net)
            add_arc_from_to(fTrans, placeF, net)
            add_arc_from_to(placeF, new_final_trans, net)

            add_arc_from_to(control_place, iTrans, net)
            add_arc_from_to(fTrans, control_place, net)

            net, message_petri_net, counts, intermediate_place = recursively_add_tree(
                subtree,
                net,
                message_petri_net,
                iTrans,
                fTrans,
                counts,
                rec_depth + 1,
                participant,
            )

    elif tree.operator == Operator.SEQUENCE:
        intermediate_place = initial_place
        for idx, child in enumerate(tree_children):
            final_connection_place = None
            if idx == len(tree_children) - 1:
                final_connection_place = final_place
            net, message_petri_net, counts, intermediate_place = recursively_add_tree(
                child,
                net,
                message_petri_net,
                intermediate_place,
                final_connection_place,
                counts,
                rec_depth + 1,
                participant,
            )
    elif tree.operator == Operator.LOOP:
        # if not parent_tree.operator == Operator.SEQUENCE:
        new_initial_place = get_new_place(counts)
        net.places.add(new_initial_place)
        init_loop_trans = get_new_hidden_trans(counts, type_trans="init_loop")
        net.transitions.add(init_loop_trans)
        add_arc_from_to(initial_place, init_loop_trans, net)
        add_arc_from_to(init_loop_trans, new_initial_place, net)
        initial_place = new_initial_place
        loop_trans = get_new_hidden_trans(counts, type_trans="loop")
        net.transitions.add(loop_trans)
        if len(tree_children) == 1:
            net, message_petri_net, counts, intermediate_place = recursively_add_tree(
                tree_children[0],
                net,
                message_petri_net,
                initial_place,
                final_place,
                counts,
                rec_depth + 1,
                participant,
            )
            add_arc_from_to(final_place, loop_trans, net)
            add_arc_from_to(loop_trans, initial_place, net)
        else:
            dummy = ProcessTree()
            do = tree_children[0]
            redo = tree_children[1]
            _exit = (
                tree_children[2]
                if len(tree_children) > 2
                and (tree_children[2].label is not None or tree_children[2].children)
                else dummy
            )

            net, message_petri_net, counts, int1 = recursively_add_tree(
                do,
                net,
                message_petri_net,
                initial_place,
                None,
                counts,
                rec_depth + 1,
                participant,
            )
            net, message_petri_net, counts, int2 = recursively_add_tree(
                redo,
                net,
                message_petri_net,
                int1,
                None,
                counts,
                rec_depth + 1,
                participant,
            )
            net, message_petri_net, counts, _ = recursively_add_tree(
                _exit,
                net,
                message_petri_net,
                int1,
                final_place,
                counts,
                rec_depth + 1,
                participant,
            )

            looping_place = int2

            add_arc_from_to(looping_place, loop_trans, net)
            add_arc_from_to(loop_trans, initial_place, net)

    return net, message_petri_net, counts, final_place


def subgraph_to_petri_net(
    tree: ProcessTree,
    message_petri_net: Optional[Dict[MessageFlow, PetriNet.Transition]] = None,
    participant: str = "petri_net",
) -> Tuple[PetriNet, Dict[MessageFlow, PetriNet.Transition], Marking, Marking]:
    """
    Convert a process tree subtree of a participant to a petri net.

    Parameters
    -----------
    tree
        The process tree.
    message_petri_net
        A dictionary of message flows to a petri net transition.
    participant
        The participant of the petri net.

    Returns
    -----------
    net
        Petri net.
    message_petri_net
        A dictionary of message flows to a petri net transition.
    initial_marking
        The initial marking in the petri net.
    final_marking
        The final marking in the petri net.
    """
    if message_petri_net is None:
        message_petri_net = {}

    counts = Counts()
    net = PetriNet(participant)
    initial_marking = Marking()
    final_marking = Marking()
    source = get_new_place(counts)
    source.name = "source"
    sink = get_new_place(counts)
    sink.name = "sink"
    net.places.add(source)
    net.places.add(sink)
    initial_marking[source] = 1
    final_marking[sink] = 1
    initial_mandatory = check_tau_mandatory_at_initial_marking(tree)
    final_mandatory = check_tau_mandatory_at_final_marking(tree)

    if initial_mandatory:
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        tau_initial = get_new_hidden_trans(counts, type_trans="tau")
        net.transitions.add(tau_initial)
        add_arc_from_to(source, tau_initial, net)
        add_arc_from_to(tau_initial, initial_place, net)
    else:
        initial_place = source

    if final_mandatory:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        tau_final = get_new_hidden_trans(counts, type_trans="tau")
        net.transitions.add(tau_final)
        add_arc_from_to(final_place, tau_final, net)
        add_arc_from_to(tau_final, sink, net)
    else:
        final_place = sink

    net, message_petri_net, counts, _ = recursively_add_tree(
        tree,
        net,
        message_petri_net,
        initial_place,
        final_place,
        counts,
        0,
        participant,
    )

    reduction.apply_simple_reduction(net)

    places = list(net.places)
    for place in places:
        if len(place.out_arcs) == 0 and place not in final_marking:
            remove_place(net, place)
        if len(place.in_arcs) == 0 and place not in initial_marking:
            remove_place(net, place)

    return net, message_petri_net, initial_marking, final_marking
