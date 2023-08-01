import unittest
import os

import pm4py

from distributed_discovery.util.read import read_xes

from distributed_discovery.discovery.im import discover_process_tree
from distributed_discovery.conversion.petri_net import process_tree_to_petri_net

import operator
from uuid import uuid4
from typing import List, Tuple, Dict, Set

from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, remove_place
from pm4py.objects.conversion.process_tree.variants.to_petri_net import (
    get_new_place,
    Counts,
)
from pm4py.objects.petri_net.obj import PetriNet, Marking

from distributed_discovery.objects.message_flow import MessageFlow


def merge_petri_nets(
    petri_nets: List[Tuple[str, PetriNet, Marking, Marking]],
    message_petri_net_per_participant: Dict[MessageFlow, PetriNet.Transition],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
):
    counts = Counts()
    net = PetriNet()
    initial_marking = Marking()
    final_marking = Marking()
    source = get_new_place(counts)
    source.name = "source1"
    sink = get_new_place(counts)
    sink.name = "sink1"
    net.places.add(source)
    net.places.add(sink)
    initial_marking[source] = 1
    final_marking[sink] = 1

    memodict = {}

    end_hidden_trans = PetriNet.Transition("end", None)
    net.transitions.add(end_hidden_trans)

    start_hidden_trans = PetriNet.Transition("start", None)
    net.transitions.add(start_hidden_trans)

    petri_nets.sort(key=operator.itemgetter(0))

    for participant, from_petri_net, im, fm in petri_nets:
        copy_values(
            from_petri_net,
            participant,
            net,
            start_hidden_trans,
            end_hidden_trans,
            memodict,
        )

    add_arc_from_to(source, start_hidden_trans, net, weight=1)
    add_arc_from_to(end_hidden_trans, sink, net, weight=1)

    add_message_flows(net, message_petri_net_per_participant, sent_messages, memodict)

    places = list(net.places)
    for place in places:
        if len(place.out_arcs) == 0 and place not in final_marking:
            remove_place(net, place)
        if len(place.in_arcs) == 0 and place not in initial_marking:
            remove_place(net, place)

    return net, initial_marking, final_marking


def add_message_flows(
    petri_net: PetriNet,
    message_petri_net_per_participant: Dict[MessageFlow, PetriNet.Transition],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
    memodict,
):
    for participant, entries in sent_messages.items():
        for sender, receivers in entries.items():
            for receiver in receivers:
                sender_transition = message_petri_net_per_participant[
                    MessageFlow(participant, sender)
                ]
                receiver_transition = message_petri_net_per_participant[receiver]

                place = PetriNet.Place(str(uuid4()))
                petri_net.places.add(place)

                add_arc_from_to(memodict[id(sender_transition)], place, petri_net)
                add_arc_from_to(place, memodict[id(receiver_transition)], petri_net)


def copy_values(
    from_net: PetriNet,
    participant,
    to_net: PetriNet,
    start: PetriNet.Transition,
    end: PetriNet.Transition,
    memodict,
):
    memodict[id(from_net)] = to_net
    for place in from_net.places:

        place_copy = PetriNet.Place(place.name, properties=place.properties)
        to_net.places.add(place_copy)
        memodict[id(place)] = place_copy
        if place.name == f"{participant}_sink":
            add_arc_from_to(place_copy, end, to_net, weight=1)
        if place.name == f"{participant}_source":
            add_arc_from_to(start, place_copy, to_net, weight=1)
    for trans in from_net.transitions:
        trans_copy = PetriNet.Transition(
            trans.name, trans.label, properties=trans.properties
        )
        to_net.transitions.add(trans_copy)
        memodict[id(trans)] = trans_copy
    for arc in from_net.arcs:
        add_arc_from_to(
            memodict[id(arc.source)],
            memodict[id(arc.target)],
            to_net,
            weight=arc.weight,
        )


def evaluate(log_path: str, collective_log_path: str, petri_net_file_path: str):
    log = read_xes(log_path)
    process_tree, sent_messages_per_participant = discover_process_tree(log)
    petri_nets, message_petri_net_per_participant = process_tree_to_petri_net(
        process_tree
    )
    net, im, fm = merge_petri_nets(
        petri_nets, message_petri_net_per_participant, sent_messages_per_participant
    )

    collective_log = pm4py.read_xes(collective_log_path)

    prec = pm4py.precision_token_based_replay(collective_log, net, im, fm)
    fitness = pm4py.fitness_token_based_replay(collective_log, net, im, fm)
    print(f"Fitness {fitness['average_trace_fitness']:.3f} Precision {prec:.3f}")

    c_net, c_im, c_fm = pm4py.read_pnml(petri_net_file_path)

    c_prec = pm4py.precision_token_based_replay(collective_log, c_net, c_im, c_fm)
    c_fitness = pm4py.fitness_token_based_replay(collective_log, c_net, c_im, c_fm)
    print(
        f"Colliery: Fitness {c_fitness['average_trace_fitness']:.3f} Precision {c_prec:.3f} "
    )


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def evaluate_scenario(self, scenario: str):
        evaluate(
            f"{self.dir}/xes/validation/{scenario}.xes",
            f"{self.dir}/colliery/{scenario}-collectivelog.xes",
            f"{self.dir}/colliery/{scenario}-petrinet-of-collaboration-discovered-inductive.pnml",
        )

    def test_healthcare(self):
        print("\n===== SCENARIO 1 - HEALTHCARE =====")
        self.evaluate_scenario("1-healthcare")

    def test_travel_agency(self):
        print("\n===== SCENARIO 2 - TRAVEL AGENCY =====")
        self.evaluate_scenario("2-travel-agency")

    def test_thermostat(self):
        print("\n===== SCENARIO 3 - THERMOSTAT =====")
        self.evaluate_scenario("3-thermostat")

    def test_zoo(self):
        print("\n===== SCENARIO 4 - ZOO =====")
        self.evaluate_scenario("4-zoo")


if __name__ == "__main__":
    unittest.main()
