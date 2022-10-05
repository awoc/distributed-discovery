import unittest
import os

from distributed_discovery.util.read import read_xes
from distributed_discovery.objects.log import DistributedTrace


class TestDistributedTraceClass(unittest.TestCase):
    def setUp(self) -> None:
        self.trace: DistributedTrace = read_xes(f"{os.path.dirname(__file__)}/xes/send-receive.xes")[0]

    def test_eq(self) -> None:
        self.assertEqual(self.trace, self.trace)
        trace2 = DistributedTrace(self.trace._list, attributes=self.trace.attributes)
        self.assertNotEqual(self.trace, trace2)

    def test_copy(self) -> None:
        trace2 = self.trace.__copy__()
        self.assertEqual(self.trace, trace2)
        # Check reference
        self.assertFalse(self.trace is trace2)
        self.assertFalse(self.trace._list is trace2._list)
        self.assertFalse(self.trace._attributes is trace2._attributes)
        # Same reference
        self.assertTrue(self.trace._list[0] is trace2._list[0])

    def test_deep_copy(self) -> None:
        trace2 = self.trace.__deepcopy__()
        self.assertEqual(self.trace, trace2)
        # Check reference
        self.assertFalse(self.trace is trace2)
        self.assertFalse(self.trace._list is trace2._list)
        self.assertFalse(self.trace._attributes is trace2._attributes)
        # Different reference
        self.assertFalse(self.trace._list[0] is trace2._list[0])


if __name__ == "__main__":
    unittest.main()
