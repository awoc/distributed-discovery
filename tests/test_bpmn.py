import unittest
import os

from distributed_discovery import discover_bpmn
from distributed_discovery.util.read import read_xes
from distributed_discovery.discovery.im import discover_process_tree
from distributed_discovery.conversion.bpmn import process_tree_to_bpmn


class TestBPMN(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            process_tree_to_bpmn("")

    def test_single_participant(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        process_tree, _ = discover_process_tree(log)
        participant1 = process_tree.children[0]

        bpmns, _ = process_tree_to_bpmn(participant1)

        self.assertEqual(1, len(bpmns))

    def test_log1(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        process_tree, _ = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)

        # 2 Participants
        self.assertEqual(2, len(bpmns))

        bpmn_per_participant = {bpmn.get_name(): bpmn for bpmn in bpmns}
        send_receive1_bpmn = bpmn_per_participant["SendReceive1"]
        send_receive2_bpmn = bpmn_per_participant["SendReceive2"]

        # 9 Nodes = Start Node + End Node + 2 Parallel Gateways + 5 Activities
        self.assertEqual(9, len(send_receive1_bpmn.get_nodes()))
        self.assertEqual(9, len(send_receive1_bpmn.get_flows()))
        self.assertEqual("SendReceive1", send_receive1_bpmn.get_name())

        self.assertEqual(3, len(send_receive2_bpmn.get_nodes()))
        self.assertEqual(2, len(send_receive2_bpmn.get_flows()))
        self.assertEqual("SendReceive2", send_receive2_bpmn.get_name())

        # 4 Sending or Receiving Activities
        self.assertEqual(4, len(message_bpmn))

    def test_discover_log1(self):
        log = read_xes(f"{self.dir}/xes/send-receive.xes")
        _ = discover_bpmn(log)

    def test_log2(self):
        log = read_xes(f"{self.dir}/xes/one-to-many.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)

        self.assertEqual(3, len(bpmns))

        bpmn_per_participant = {bpmn.get_name(): bpmn for bpmn in bpmns}
        one_to_many1_bpmn = bpmn_per_participant["OneToMany1"]
        one_to_many2_bpmn = bpmn_per_participant["OneToMany2"]
        one_to_many3_bpmn = bpmn_per_participant["OneToMany3"]

        self.assertEqual(13, len(one_to_many1_bpmn.get_nodes()))
        self.assertEqual(14, len(one_to_many1_bpmn.get_flows()))
        self.assertEqual("OneToMany1", one_to_many1_bpmn.get_name())

        self.assertEqual(5, len(one_to_many2_bpmn.get_nodes()))
        self.assertEqual(4, len(one_to_many2_bpmn.get_flows()))
        self.assertEqual("OneToMany2", one_to_many2_bpmn.get_name())

        self.assertEqual(5, len(one_to_many3_bpmn.get_nodes()))
        self.assertEqual(4, len(one_to_many3_bpmn.get_flows()))
        self.assertEqual("OneToMany3", one_to_many3_bpmn.get_name())

        self.assertEqual(7, len(message_bpmn))

    def test_log3(self):
        log = read_xes(f"{self.dir}/xes/stream.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)

        self.assertEqual(2, len(bpmns))

        bpmn_per_participant = {bpmn.get_name(): bpmn for bpmn in bpmns}
        stream1_bpmn = bpmn_per_participant["Stream1"]
        stream2_bpmn = bpmn_per_participant["Stream2"]

        self.assertEqual(9, len(stream1_bpmn.get_nodes()))
        self.assertEqual(9, len(stream1_bpmn.get_flows()))
        self.assertEqual("Stream1", stream1_bpmn.get_name())

        self.assertEqual(9, len(stream2_bpmn.get_nodes()))
        self.assertEqual(9, len(stream2_bpmn.get_flows()))
        self.assertEqual("Stream2", stream2_bpmn.get_name())

        self.assertEqual(4, len(message_bpmn))

    def test_log4(self):
        log = read_xes(f"{self.dir}/xes/supply-chain.xes")

        process_tree, sent_messages_per_participant = discover_process_tree(log)
        bpmns, message_bpmn = process_tree_to_bpmn(process_tree)

        self.assertEqual(5, len(bpmns))

        bpmn_per_participant = {bpmn.get_name(): bpmn for bpmn in bpmns}
        supplier_c_bpmn = bpmn_per_participant["SupplierC"]
        supplier_a_bpmn = bpmn_per_participant["SupplierA"]
        customer_bpmn = bpmn_per_participant["Customer"]

        self.assertEqual(4, len(supplier_c_bpmn.get_nodes()))
        self.assertEqual(3, len(supplier_c_bpmn.get_flows()))
        self.assertEqual("SupplierC", supplier_c_bpmn.get_name())

        self.assertEqual(6, len(supplier_a_bpmn.get_nodes()))
        self.assertEqual(5, len(supplier_a_bpmn.get_flows()))
        self.assertEqual("SupplierA", supplier_a_bpmn.get_name())

        self.assertEqual(5, len(customer_bpmn.get_nodes()))
        self.assertEqual(4, len(customer_bpmn.get_flows()))
        self.assertEqual("Customer", customer_bpmn.get_name())

        self.assertEqual(22, len(message_bpmn))


if __name__ == "__main__":
    unittest.main()
