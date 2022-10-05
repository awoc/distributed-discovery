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
from typing import List, Dict, Tuple, Set

from pm4py.objects.bpmn.obj import BPMN

from distributed_discovery.discovery.im import discover_process_tree
from distributed_discovery.conversion.petri_net import process_tree_to_petri_net
from distributed_discovery.conversion.bpmn import process_tree_to_bpmn
from distributed_discovery.objects.log import DistributedEventLog
from distributed_discovery.objects.message_flow import MessageFlow


def discover_bpmn(
    log: DistributedEventLog, noise_threshold: float = 0.0
) -> Tuple[
    List[BPMN],
    Dict[MessageFlow, List[BPMN.Event]],
    Dict[str, Dict[str, Set[MessageFlow]]],
]:
    """
    Discovers a BPMN collaboration diagram using the inductive miner.

    Parameters
    ----------
    log
        The event log.
    noise_threshold
        Noise threshold (default: 0.0).

    Returns
    -------
    bpmn_graphs
        A list of converted BPMN graphs.
    message_bpmn
        Dictionary of activities with BPMN Events.
    sent_messages
        A dictionary of message flows per participant.
    """
    process_tree, sent_messages = discover_process_tree(log, noise_threshold)
    bpmn_graphs, message_bpmn = process_tree_to_bpmn(process_tree)
    return bpmn_graphs, message_bpmn, sent_messages


def discover_petri_net(log: DistributedEventLog, noise_threshold: float = 0.0):
    """
    Discovers a petri net using the inductive miner.

    Parameters
    ----------
    log
        The event log.
    noise_threshold
        Noise threshold (default: 0.0).

    Returns
    -------
    petri_nets
        A list of petri nets per participant.
    message_petri_net
         A dictionary of message flows to a petri net transition.
    """
    process_tree, _ = discover_process_tree(log, noise_threshold)
    return process_tree_to_petri_net(process_tree)
