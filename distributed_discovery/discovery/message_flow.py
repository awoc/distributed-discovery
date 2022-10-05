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
from typing import Tuple, List
from itertools import zip_longest
import operator

from distributed_discovery.objects.log import DistributedEventLog
from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.util.filter import get_message_events


def discover_message_flows(
    log: DistributedEventLog,
) -> List[Tuple[MessageFlow, MessageFlow]]:
    """
    Discovers message flows in a distributed event log.

    Parameters
    ----------
    log
        The distributed event log.

    Returns
    -------
    message_activities
        A list of send and receive tuples.
    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    sent_message_events, received_message_events = get_message_events(log)

    message_activities = []

    intersect = set(sent_message_events.keys()).intersection(
        set(received_message_events.keys())
    )

    for message in intersect:
        # Sort activities by timestamp for correct activity association
        # if the same message is sent in multiple activities
        sent_messages = sorted(
            sent_message_events[message], key=operator.attrgetter("timestamp")
        )
        received_messages = sorted(
            received_message_events[message], key=operator.attrgetter("timestamp")
        )

        for sent, received in zip_longest(sent_messages, received_messages):
            if (
                sent is not None
                and received is not None
                and sent.participant != received.participant
            ):
                message_activities.append(
                    (
                        MessageFlow(sent.participant, sent.name),
                        MessageFlow(received.participant, received.name),
                    )
                )
            else:
                message_activities.append(
                    (
                        MessageFlow(sent.participant, sent.name)
                        if sent is not None
                        else None,
                        MessageFlow(received.participant, received.name)
                        if received is not None
                        else None,
                    )
                )

    return message_activities


def discover_performance_message_flows(log: DistributedEventLog):
    """
    Discovers message flows in a distributed event log with performance information.
    Only successfully messages are considered.

    Parameters
    ----------
    log
        The distributed event log.

    Returns
    -------
    message_activities
        A list of send and receive tuples.
    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    sent_message_activities, received_message_activities = get_message_events(log)

    message_activities = []

    intersect = set(sent_message_activities.keys()).intersection(
        set(received_message_activities.keys())
    )

    for message in intersect:
        # Sort activities by timestamp for correct activity association
        # if the same message is sent in multiple activities
        sent_messages = sorted(
            sent_message_activities[message], key=operator.attrgetter("timestamp")
        )
        received_messages = sorted(
            received_message_activities[message], key=operator.attrgetter("timestamp")
        )

        for sent, received in zip_longest(sent_messages, received_messages):
            if (
                sent is not None
                and received is not None
                and sent.participant != received
            ):
                message_activities.append(
                    (
                        MessageFlow(sent.participant, sent.name),
                        MessageFlow(received.participant, received.name),
                        max(0, (sent.timestamp - received.timestamp).total_seconds()),
                    )
                )

    return message_activities
