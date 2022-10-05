import unittest
import os

from distributed_discovery.util.read import read_xes
from distributed_discovery.discovery.im import discover_process_tree
from distributed_discovery.visualization.process_tree import visualize_process_tree


class TestProcessTree(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            visualize_process_tree("")

    def test_log_send_receive(self) -> None:
        log = read_xes(f"{self.dir}/xes/send-receive.xes")
        process_tree, _ = discover_process_tree(log)
        _ = visualize_process_tree(process_tree)

    def test_log_healthcare(self):
        log = read_xes(f"{self.dir}/xes/validation/1-healthcare.xes")
        process_tree, _ = discover_process_tree(log)
        _ = visualize_process_tree(process_tree)


if __name__ == "__main__":
    unittest.main()
