import unittest
import os
import tempfile

from distributed_discovery.discovery.im import discover_process_tree
from distributed_discovery.conversion.bpmn import process_tree_to_bpmn
from distributed_discovery.util.read import read_xes
from distributed_discovery.export.bpmn import write_bpmn


class TestBPMN(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_log_send_receive(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_choice(self):
        log = read_xes(f"{self.dir}/xes/choice.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_one_from_many(self):
        log = read_xes(f"{self.dir}/xes/one-from-many.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_one_to_many_send(self):
        log = read_xes(f"{self.dir}/xes/one-to-many-send.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_stream(self):
        log = read_xes(f"{self.dir}/xes/stream.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_supply_chain(self):
        log = read_xes(f"{self.dir}/xes/supply-chain.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_healthcare(self):
        log = read_xes(f"{self.dir}/xes/validation/1-healthcare.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_travel_agency(self):
        log = read_xes(f"{self.dir}/xes/validation/2-travel-agency.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_thermostat(self):
        log = read_xes(f"{self.dir}/xes/validation/3-thermostat.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)

    def test_log_zoo(self):
        log = read_xes(f"{self.dir}/xes/validation/4-zoo.xes")
        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)

        with tempfile.NamedTemporaryFile() as f:
            write_bpmn(f.name, bpmn_graphs, message_bpmn, sent_messages_per_participant)


if __name__ == "__main__":
    unittest.main()
