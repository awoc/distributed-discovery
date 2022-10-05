import unittest
import os

from distributed_discovery.util.read import read_xes


class TestXESImport(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_log1(self) -> None:
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        expected_participants = {"SendReceive1", "SendReceive2"}

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(16, len(log))

    def test_log2(self) -> None:
        log = read_xes(f"{self.dir}/xes/choice.xes")

        expected_participants = {"choice1", "choice2", "choice3"}

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(10, len(log))

    def test_log3(self) -> None:
        log = read_xes(f"{self.dir}/xes/one-from-many.xes")

        expected_participants = {"OneFromMany1", "OneFromMany2", "OneFromMany3"}

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(12, len(log))

    def test_log4(self) -> None:
        log = read_xes(f"{self.dir}/xes/one-to-many-send.xes")

        expected_participants = {"OneToManySend1", "OneToManySend2", "OneToManySend3"}

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(12, len(log))

    def test_log5(self) -> None:
        log = read_xes(f"{self.dir}/xes/one-to-many.xes")

        expected_participants = {"OneToMany1", "OneToMany2", "OneToMany3"}

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(18, len(log))

    def test_log6(self) -> None:
        log = read_xes(f"{self.dir}/xes/stream.xes")

        expected_participants = {"Stream1", "Stream2"}

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(10, len(log))

    def test_log7(self) -> None:
        log = read_xes(f"{self.dir}/xes/supply-chain.xes")

        expected_participants = {
            "Customer",
            "Manufacturer",
            "SupplierA",
            "SupplierB",
            "SupplierC",
        }

        self.assertCountEqual(log.participants, expected_participants)
        self.assertSetEqual(log.participants, expected_participants)

        self.assertEqual(46, len(log))


if __name__ == "__main__":
    unittest.main()
