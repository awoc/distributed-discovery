import unittest
import os

from distributed_discovery.util.read import read_xes
from distributed_discovery.discovery.dfg import discover_dfg
from distributed_discovery.objects.message_flow import MessageFlow


class TestDFG(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            discover_dfg("")

    def test_log_send_receive(self) -> None:
        log = read_xes(f"{self.dir}/xes/send-receive.xes")

        dfg = discover_dfg(log)

        len(dfg.participant_dfgs)
        # Number of participants must match
        self.assertEqual(len(log.participants), len(dfg.participant_dfgs))
        # 2 message flows expected
        self.assertEqual(2, len(dfg.messages_dfg))
        # Expect 6 message exchanges
        self.assertEqual(6, list(dfg.messages_dfg.values())[0])
        self.assertEqual(6, list(dfg.messages_dfg.values())[1])
        # Check sending/receiving activities
        self.check_message_flow(
            MessageFlow("SendReceive1", "Send Order"),
            MessageFlow("SendReceive2", "Receive Order"),
            dfg.messages_dfg,
        )
        self.check_message_flow(
            MessageFlow("SendReceive2", "Send Receipt"),
            MessageFlow("SendReceive1", "Receive Receipt"),
            dfg.messages_dfg,
        )
        # Check number of transitions without start and end activity
        self.assertEqual(4, len(dfg.participant_dfgs["SendReceive1"]))
        self.assertEqual(2, len(dfg.participant_dfgs["SendReceive2"]))

    def check_message_flow(
        self,
        message_flow1: MessageFlow,
        message_flow2: MessageFlow,
        messages_dfg: dict[tuple[MessageFlow, MessageFlow], int],
    ) -> None:
        message_flow = (message_flow1, message_flow2)
        self.assertIn(message_flow, messages_dfg)


if __name__ == "__main__":
    unittest.main()
