"""
    Part of distributed discovery project.
    Copyright (C) 2022  Alexander Collins

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from typing import List, Dict, Set, Tuple

from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.petri_net.obj import PetriNet, Marking

from distributed_discovery.objects.dfg import DistributedDfg
from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.visualization.bpmn import visualize_bpmn
from distributed_discovery.visualization.dfg import visualize_dfg
from distributed_discovery.visualization.process_tree import visualize_process_tree
from distributed_discovery.visualization.petri_net import visualize_petri_net
from distributed_discovery.objects.process_tree import ProcessTree


def view_dfg(dfg: DistributedDfg) -> None:
    """
    Views a DFG.

    Parameters
    ----------
    dfg
        The DFG.

    Returns
    -------

    """
    visualize_dfg(dfg).view()


def view_process_tree(process_tree: ProcessTree) -> None:
    """
    Views a process tree.

    Parameters
    ----------
    process_tree
        The process tree.

    Returns
    -------

    """
    visualize_process_tree(process_tree).view()


def view_bpmn(
    bpmn_graphs: List[BPMN],
    message_bpmn: Dict[MessageFlow, List[BPMN.BPMNNode]],
    sent_messages_per_participant: Dict[str, Dict[str, Set[MessageFlow]]],
) -> None:
    """
    Views a collaboration BPMN.

    Parameters
    ----------
    bpmn_graphs:
        List of BPMN graphs. One per participant.
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.
    sent_messages_per_participant
        A dictionary of message flows per participant.

    Returns
    -------

    """
    visualize_bpmn(bpmn_graphs, message_bpmn, sent_messages_per_participant).view()


def view_petri_net(
    petri_nets: List[Tuple[str, PetriNet, Marking, Marking]],
    message_petri_net_per_participant: Dict[MessageFlow, PetriNet.Transition],
    sent_messages_per_participant: Dict[str, Dict[str, Set[MessageFlow]]],
) -> None:
    """
    Views a petri net.

    Parameters
    ----------
    petri_nets
        List of petri nets of each participant with start and end markings.
    message_petri_net_per_participant
        A dictionary of message flows to a petri net transition.
    sent_messages_per_participant
        Sent messages per participant.

    Returns
    -------

    """
    visualize_petri_net(
        petri_nets, message_petri_net_per_participant, sent_messages_per_participant
    ).view()
