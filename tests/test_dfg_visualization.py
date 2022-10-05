import unittest
import os

from distributed_discovery.util.read import read_xes
from distributed_discovery.visualization.dfg import visualize_dfg
from distributed_discovery.discovery.dfg import discover_dfg
from distributed_discovery.objects.dfg import DfgType


class TestDFGVisualization(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_log_performance(self) -> None:
        log = read_xes(f"{self.dir}/xes/choice.xes")

        dfg = discover_dfg(log, variant=DfgType.PERFORMANCE)
        _ = visualize_dfg(dfg)

    def test_log_send_receive(self):
        # Test correct execution
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        dfg = discover_dfg(log)
        _ = visualize_dfg(dfg)

    def test_log_duplicate_message_log(self) -> None:
        log = read_xes(f"{self.dir}/xes/one-to-many-duplicate-messages.xes")

        dfg = discover_dfg(log)
        visualize_dfg(dfg)


if __name__ == "__main__":
    unittest.main()
