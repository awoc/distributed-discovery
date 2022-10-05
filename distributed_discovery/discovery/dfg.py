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
from typing import Optional
from collections import Counter

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

from distributed_discovery.objects.log import DistributedEventLog
from distributed_discovery.objects.dfg import DistributedDfg, DfgType
from distributed_discovery.discovery.message_flow import (
    discover_message_flows,
    discover_performance_message_flows,
)
from distributed_discovery.util.filter import get_participants_event_logs


def discover_dfg(
    log: DistributedEventLog, variant: Optional[DfgType] = DfgType.FREQUENCY
) -> DistributedDfg:
    if variant == variant.FREQUENCY:
        return discover_frequency_dfg(log)
    else:
        return discover_performance_dfg(log)


def discover_frequency_dfg(log: DistributedEventLog) -> DistributedDfg:
    """
    Discovers a directly-follows graph from a distributed event log including message flows.

    Parameters
    ----------
    log
        The distributed event log.

    Returns
    -------
    dfg
        The discovered directly-follows graph.
    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    dfg_participants = {}

    for participant, event_log in get_participants_event_logs(log).items():
        dfg = dfg_discovery.apply(event_log)
        dfg_participants[participant] = dfg

    exchanged_messages = discover_message_flows(log)
    dfg_messages = Counter(exchanged_messages)

    return DistributedDfg(dfg_messages, dfg_participants, log, DfgType.FREQUENCY)


def discover_performance_dfg(log: DistributedEventLog) -> DistributedDfg:
    """
    Discovers a directly-follows graph from a distributed event log including message flows.

    Parameters
    ----------
    log
        The distributed event log.

    Returns
    -------
    dfg
        The discovered directly-follows graph.
    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    dfg_participants = {}

    for participant, event_log in get_participants_event_logs(log).items():
        dfg = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.PERFORMANCE)
        dfg_participants[participant] = dfg

    exchanged_messages = discover_performance_message_flows(log)

    dfg_message_flow_count = {}
    dfg_message_flow_duration = {}

    for sender, receiver, duration in exchanged_messages:
        if (sender, receiver) in dfg_message_flow_count:
            dfg_message_flow_count[(sender, receiver)] += 1
            dfg_message_flow_duration[(sender, receiver)] += duration
        else:
            dfg_message_flow_count[(sender, receiver)] = 1
            dfg_message_flow_duration[(sender, receiver)] = duration

    dfg_messages = {
        message_flow: (
            (total_count / dfg_message_flow_duration[message_flow])
            if dfg_message_flow_duration[message_flow] > 0
            else 0
        )
        for message_flow, total_count in dfg_message_flow_count.items()
    }

    return DistributedDfg(dfg_messages, dfg_participants, log, DfgType.PERFORMANCE)
