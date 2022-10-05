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
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from distributed_discovery.objects.log import DistributedEventLog
from distributed_discovery.objects.message_flow import MessageFlow


class DfgType(Enum):
    """
    Enum of DFG types.
    """

    PERFORMANCE = "performance"
    FREQUENCY = "frequency"


@dataclass
class DistributedDfg:
    """
    Used to store information for the visualization of DFGs.
    """

    messages_dfg: Dict[Tuple[MessageFlow, MessageFlow], int]
    participant_dfgs: Dict[str, Dict[Tuple[str, str], float]]
    event_log: DistributedEventLog
    type: DfgType
