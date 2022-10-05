import unittest
import os

from distributed_discovery.util.read import read_xes
from distributed_discovery.objects.log import DistributedEventLog


class TestDistributedEventLogClass(unittest.TestCase):
    def setUp(self) -> None:
        self.event_log: DistributedEventLog = read_xes(f"{os.path.dirname(__file__)}/xes/send-receive.xes")

    def test_eq(self) -> None:
        self.assertEqual(self.event_log, self.event_log)

    def test_copy(self) -> None:
        event_log2 = self.event_log.__copy__()
        self.assertEqual(self.event_log, event_log2)
        # Check reference
        self.assertFalse(self.event_log is event_log2)
        self.assertFalse(self.event_log._list is event_log2._list)
        self.assertFalse(self.event_log._attributes is event_log2._attributes)
        self.assertFalse(self.event_log.participants is event_log2.participants)
        self.assertFalse(self.event_log.extensions is event_log2.extensions)
        self.assertFalse(self.event_log.classifiers is event_log2.classifiers)
        self.assertFalse(self.event_log.omni_present is event_log2.omni_present)
        # Same reference
        self.assertTrue(self.event_log._list[0] is event_log2._list[0])

    def test_deep_copy(self) -> None:
        event_log2 = self.event_log.__deepcopy__()
        self.assertEqual(self.event_log, event_log2)
        # Check reference
        self.assertFalse(self.event_log is event_log2)
        self.assertFalse(self.event_log._list is event_log2._list)
        self.assertFalse(self.event_log._attributes is event_log2._attributes)
        self.assertFalse(self.event_log.participants is event_log2.participants)
        self.assertFalse(self.event_log.extensions is event_log2.extensions)
        self.assertFalse(self.event_log.classifiers is event_log2.classifiers)
        self.assertFalse(self.event_log.omni_present is event_log2.omni_present)
        # Difference reference
        self.assertFalse(self.event_log._list[0] is event_log2._list[0])


if __name__ == "__main__":
    unittest.main()
