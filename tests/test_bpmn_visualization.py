import unittest
import os

from distributed_discovery.util.read import read_xes
from distributed_discovery.discovery.im import discover_process_tree
from distributed_discovery.conversion.bpmn import process_tree_to_bpmn
from distributed_discovery.visualization.bpmn import visualize_bpmn


class TestBPMN(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            visualize_bpmn([("", "")], {}, {})

    def test_single_participant(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        process_tree, _ = discover_process_tree(log)
        participant1 = process_tree.children[0]

        bpmns, message_bpmn = process_tree_to_bpmn(participant1)
        visualize_bpmn(bpmns, message_bpmn, {}).view()

    def test_log1(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)
        _ = visualize_bpmn(bpmns, message_bpmn, sent_messages_per_participant)

    def test_log2(self):
        log = read_xes(f"{self.dir}/xes/one-to-many.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)
        _ = visualize_bpmn(bpmns, message_bpmn, sent_messages_per_participant)

    def test_log3(self):
        log = read_xes(f"{self.dir}/xes/stream.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)
        _ = visualize_bpmn(bpmns, message_bpmn, sent_messages_per_participant)

    def test_log4(self):
        log = read_xes(f"{self.dir}/xes/supply-chain.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)
        _ = visualize_bpmn(bpmns, message_bpmn, sent_messages_per_participant)

    def test_log5(self):
        log = read_xes(f"{self.dir}/xes/one-to-many-multiple-receiving-messages.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)
        _ = visualize_bpmn(bpmns, message_bpmn, sent_messages_per_participant)


if __name__ == "__main__":
    unittest.main()
