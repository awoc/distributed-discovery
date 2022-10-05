import unittest
import os
from typing import List, Tuple

from distributed_discovery.objects.process_tree import ProcessTree
from distributed_discovery.util.read import read_xes
from distributed_discovery.discovery.im import (
    discover_process_tree,
    discover_sent_received_messages_per_participant,
)
from distributed_discovery.objects.process_tree_operator import Operator


class TestProcessTree(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            discover_process_tree("")

    def test_wrong_input2(self):
        with self.assertRaises(TypeError):
            discover_sent_received_messages_per_participant("")

    def test_log_send_receive(self) -> None:
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        # 2 Participants per subgraph
        self.assertEqual(2, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertIn("SendReceive1", labels)
        self.assertIn("SendReceive2", labels)

        first_subgraph_message_flows = self.process_tree_find_message_flows(
            process_tree.children[0]
        )
        second_subgraph_message_flows = self.process_tree_find_message_flows(
            process_tree.children[1]
        )

        self.assertListEqual(
            first_subgraph_message_flows, second_subgraph_message_flows
        )
        self.assertEqual(2, len(first_subgraph_message_flows))

        sending_activities2 = [pair[0] for pair in first_subgraph_message_flows]
        receiving_activities2 = [pair[1] for pair in first_subgraph_message_flows]

        self.assertListEqual(["Send Order", "Send Receipt"], sending_activities2)
        self.assertListEqual(
            ["Receive Order", "Receive Receipt"], receiving_activities2
        )

    def test_discover_log_send_receive(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")
        _ = discover_process_tree(log)

    def test_log_choice(self):
        log = read_xes(f"{self.dir}/xes/choice.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(3, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["choice1", "choice2", "choice3"], labels)

        self.check_participant_in_process_tree(
            process_tree, "choice1", ["Send to 3", "Send to 2"],
        )
        self.check_participant_in_process_tree(
            process_tree, "choice2", ["Send to 2"],
        )

        self.check_participant_in_process_tree(
            process_tree, "choice3", ["Send to 3"],
        )

    def test_log_one_from_many(self):
        log = read_xes(f"{self.dir}/xes/one-from-many.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(3, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["OneFromMany1", "OneFromMany2", "OneFromMany3"], labels)

        self.check_participant_in_process_tree(
            process_tree, "OneFromMany1", ["Send Receipt", "Send Order"],
        )
        self.check_participant_in_process_tree(
            process_tree, "OneFromMany2", ["Send Order"],
        )

        self.check_participant_in_process_tree(
            process_tree, "OneFromMany3", ["Send Receipt"],
        )

    def test_log_one_to_many_send(self):
        log = read_xes(f"{self.dir}/xes/one-to-many-send.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(3, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(
            ["OneToManySend1", "OneToManySend2", "OneToManySend3"], labels
        )

        self.check_participant_in_process_tree(
            process_tree, "OneToManySend1", ["Send Messages", "Send Messages"],
        )
        self.check_participant_in_process_tree(
            process_tree, "OneToManySend2", ["Send Messages"],
        )

        self.check_participant_in_process_tree(
            process_tree, "OneToManySend3", ["Send Messages"],
        )

    def test_log_one_to_many(self):
        log = read_xes(f"{self.dir}/xes/one-to-many.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(3, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["OneToMany1", "OneToMany2", "OneToMany3"], labels)

        self.check_participant_in_process_tree(
            process_tree, "OneToMany1", ["Send", "Send", "Send", "Send Stream"],
        )
        self.check_participant_in_process_tree(
            process_tree, "OneToMany2", ["Send", "Send"],
        )

        self.check_participant_in_process_tree(
            process_tree, "OneToMany3", ["Send", "Send Stream"],
        )

    def test_log_stream(self):
        log = read_xes(f"{self.dir}/xes/stream.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(2, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["Stream1", "Stream2"], labels)

        self.check_participant_in_process_tree(
            process_tree, "Stream1", ["Initiate", "Send Stream"],
        )
        self.check_participant_in_process_tree(
            process_tree, "Stream2", ["Initiate", "Send Stream"],
        )

    def test_log_supply_chain(self):
        log = read_xes(f"{self.dir}/xes/supply-chain.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(5, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(
            ["SupplierA", "SupplierB", "SupplierC", "Manufacturer", "Customer"], labels
        )

        self.check_participant_in_process_tree(
            process_tree,
            "SupplierA",
            ["Order Product A", "Send Order Status A", "Deliver Product A"],
        )
        self.check_participant_in_process_tree(
            process_tree,
            "SupplierB",
            [
                "Order Part C",
                "Deliver Part C",
                "Order Part B",
                "Send Order Status B",
                "Deliver Part B",
            ],
        )

        self.check_participant_in_process_tree(
            process_tree, "SupplierC", ["Order Part C", "Deliver Part C"],
        )

        self.check_participant_in_process_tree(
            process_tree,
            "Manufacturer",
            [
                "Order Product",
                "Send Order Status Report",
                "Deliver Product",
                "Order Part B",
                "Send Order Status B",
                "Deliver Part B",
                "Order Product A",
                "Send Order Status A",
                "Deliver Product A",
            ],
        )

        self.check_participant_in_process_tree(
            process_tree,
            "Customer",
            ["Order Product", "Send Order Status Report", "Deliver Product"],
        )

    def test_log_healthcare(self):
        log = read_xes(f"{self.dir}/xes/validation/1-healthcare.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(4, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(
            ["Laboratory", "Hospital", "Gynecologist", "Patient"], labels
        )

        self.check_participant_in_process_tree(
            process_tree, "Laboratory", ["Send results", "Send blood sample"],
        )
        self.check_participant_in_process_tree(
            process_tree,
            "Hospital",
            [
                "Send admission info",
                "Forward patient analysis results",
                "Require patient admission",
            ],
        )

        self.check_participant_in_process_tree(
            process_tree,
            "Gynecologist",
            [
                "Forward patient analysis results",
                "Require patient admission",
                "Send results",
                "Send blood sample",
                "Communicate the need to hospitalise",
                "Send prescription",
                "Communicate disease",
            ],
        )

        self.check_participant_in_process_tree(
            process_tree,
            "Patient",
            [
                "Send admission info",
                "Communicate the need to hospitalise",
                "Send prescription",
                "Communicate disease",
            ],
        )

    def test_log_travel_agency(self):
        log = read_xes(f"{self.dir}/xes/validation/2-travel-agency.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(2, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["Customer", "TravelAgency"], labels)

        self.check_participant_in_process_tree(
            process_tree,
            "Customer",
            [
                "Make Travel Offer",
                "Book Travel",
                "Pay Travel",
                "Order Ticket",
                "Confirm Booking",
            ],
        )
        self.check_participant_in_process_tree(
            process_tree,
            "TravelAgency",
            [
                "Make Travel Offer",
                "Book Travel",
                "Pay Travel",
                "Order Ticket",
                "Confirm Booking",
            ],
        )

    def test_log_thermostat(self):
        log = read_xes(f"{self.dir}/xes/validation/3-thermostat.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(3, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["User", "Thermostat", "Controller"], labels)

        self.check_participant_in_process_tree(
            process_tree,
            "User",
            ["Set Up Thermostat", "Send Temperature", "Switch Off Thermostat"],
        )
        self.check_participant_in_process_tree(
            process_tree,
            "Thermostat",
            [
                "Set Up Thermostat",
                "Send Temperature",
                "Switch Off Thermostat",
                "Turn On Controller",
                "Turn Off",
                "Send Env Info",
            ],
        )
        self.check_participant_in_process_tree(
            process_tree,
            "Controller",
            ["Turn On Controller", "Turn Off", "Send Env Info"],
        )

    def test_log_zoo(self):
        log = read_xes(f"{self.dir}/xes/validation/4-zoo.xes")

        process_tree, _ = discover_process_tree(log)
        self.assertEqual(Operator.PARTICIPANT, process_tree.operator)

        self.assertEqual(3, len(process_tree.children))

        labels = [sub_tree.label for sub_tree in process_tree.children]
        self.assertCountEqual(["Zoo", "Visitor", "Bank"], labels)

        self.check_participant_in_process_tree(
            process_tree,
            "Zoo",
            [
                "Send payment request",
                "Change account",
                "Deliver Zooclub card",
                "Send information to the ZooClub department",
            ],
        )
        self.check_participant_in_process_tree(
            process_tree,
            "Visitor",
            ["Send information to the ZooClub department", "Deliver Zooclub card"],
        )

        self.check_participant_in_process_tree(
            process_tree, "Bank", ["Send payment request", "Change account"],
        )

    def check_participant_in_process_tree(
        self,
        process_tree: ProcessTree,
        participant: str,
        expected_activities: List[str],
    ):
        labels = [sub_tree.label for sub_tree in process_tree.children]
        idx = labels.index(participant)

        message_flows = self.process_tree_find_message_flows(process_tree.children[idx])
        self.assertEqual(len(expected_activities), len(message_flows))

        sending_activities = [pair[0] for pair in message_flows]

        self.assertCountEqual(expected_activities, sending_activities)

    def process_tree_find_message_flows(
        self, process_tree: ProcessTree, message_flows: List[Tuple[str, str]] = None
    ) -> List[Tuple[str, str]]:
        """
        Recursively finds all Sent and Received operators in a process tree
        and returns a pair of sending and receiving activities.

        Parameters
        ----------
        process_tree:
            A process tree.
        message_flows:
            Pairs of sending and receiving activities.
            Can be None when calling the function.

        Returns
        -------
        message_flows:
            Pairs of sending and receiving activities found in the process tree.
        """
        if message_flows is None:
            message_flows = []

        if process_tree.operator == Operator.SENT:
            sending_activity = process_tree.children[0].label
            for receiving_activity in process_tree.children[1:]:
                message_flows.append((sending_activity, receiving_activity.label))

            return message_flows

        if process_tree.operator == Operator.RECEIVED:
            receiving_activity = process_tree.children[0].label
            for sending_activity in process_tree.children[1:]:
                message_flows.append((sending_activity.label, receiving_activity))

            return message_flows

        for child in process_tree.children:
            self.process_tree_find_message_flows(child, message_flows)

        return message_flows


if __name__ == "__main__":
    unittest.main()
