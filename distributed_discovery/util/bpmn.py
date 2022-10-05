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
from typing import Dict, List

from pm4py.objects.bpmn.obj import BPMN

from distributed_discovery.objects.message_flow import MessageFlow


def copy_message_bpmn(
    message_bpmn: Dict[MessageFlow, List[BPMN.BPMNNode]]
) -> Dict[MessageFlow, List[BPMN.BPMNNode]]:
    """
    Creates a shallow copy for message flows and BPMN nodes but a deep copy for the list of BPMN nodes.

    Parameters
    ----------
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.

    Returns
    -------
    new_dict
        The copied dictionary.
    """
    new_dict = {}

    for message_flow, nodes in message_bpmn.items():
        new_dict[message_flow] = list(nodes)

    return new_dict
