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
from typing import Union, Dict, Tuple, List

from pm4py.util import xes_constants
from pm4py.objects.log.obj import EventLog, Event

from distributed_discovery.objects.log import DistributedEventLog
from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.objects.message_activity import MessageActivity
from distributed_discovery.util.constants import (
    DEFAULT_MESSAGE_SENT_KEY,
    DEFAULT_MESSAGE_RECEIVED_KEY,
)


def get_participants_event_logs(log: DistributedEventLog) -> Dict[str, EventLog]:
    """
    Generates a dictionary of every participants' event log in a distributed event log.

    Parameters
    ----------
    log
        The distributed event log.

    Returns
    -------
    event_logs
        The generated dictionary event logs per participant.
    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    event_logs: Dict[str, EventLog] = {}

    for participant in log.participants:
        participant_log = [trace for trace in log if trace.participant == participant]

        event_logs[participant] = EventLog(
            participant_log,
            attributes=log.attributes,
            extensions=log.extensions,
            classifiers=log.classifiers,
            omni_present=log.omni_present,
            properties=log.properties,
        )

    return event_logs


def get_message_events(
    log: Union[DistributedEventLog, EventLog]
) -> Tuple[Dict[str, List[MessageActivity]], Dict[str, List[MessageActivity]]]:
    """
    Finds all events that send or receive messages and stores the message content with the event in a list.
    `MessageActivity`s are stored in a list to account for events that send or receive multiple messages or
    the identical message that was sent multiple times.

    Parameters
    ----------
    log
        A `DistributedEventLog` or standard `EventLog`.

    Returns
    -------
    sent_messages
        All sent messages with their activity.
    received_messages
        All received messages with their activity.
    """
    sent_message_activities: Dict[str, List[MessageActivity]] = {}
    received_message_activities: Dict[str, List[MessageActivity]] = {}

    for trace in log:
        for event in trace:
            if DEFAULT_MESSAGE_SENT_KEY in event:
                generate_message_event(
                    event,
                    trace.participant,
                    DEFAULT_MESSAGE_SENT_KEY,
                    sent_message_activities,
                )
            elif DEFAULT_MESSAGE_RECEIVED_KEY in event:
                generate_message_event(
                    event,
                    trace.participant,
                    DEFAULT_MESSAGE_RECEIVED_KEY,
                    received_message_activities,
                )

    return sent_message_activities, received_message_activities


def generate_message_event(
    event: Event,
    participant: str,
    message_key: str,
    messages_dict: Dict[str, List[MessageActivity]],
) -> None:
    """
    Adds message to its message activities to a dictionary in-place.

    Parameters
    ----------
    event
        The event.
    participant
        The participant of the event.
    message_key
        The message key used in the event log.
    messages_dict
        The dictionary to add

    Returns
    -------

    """
    message_activity = MessageActivity(
        instance=event[xes_constants.DEFAULT_INSTANCE_KEY]
        if xes_constants.DEFAULT_INSTANCE_KEY in event
        else None,
        name=event[xes_constants.DEFAULT_NAME_KEY],
        participant=participant,
        timestamp=event[xes_constants.DEFAULT_TIMESTAMP_KEY],
    )

    for message in event[message_key]:
        if message in messages_dict:
            messages_dict[message].append(message_activity)
        else:
            messages_dict[message] = [message_activity]


def filter_out_dropped_messages(
    message_flows: List[Tuple[MessageFlow, MessageFlow]]
) -> List[Tuple[MessageFlow, MessageFlow]]:
    """
    Filters out dropped messages in a list of message flows.

    Parameters
    ----------
    message_flows
        A list of tuples of sender and receiver activities.

    Returns
    -------
    message_flows
        The filtered list of message flows.
    """
    return [
        (sent, received)
        for sent, received in message_flows
        if sent is not None and received is not None
    ]
