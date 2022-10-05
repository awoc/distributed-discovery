"""
    This file contains modified code from PM4Py.

    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
"""
from typing import Tuple, Dict, Set, Optional, List

from pm4py.algo.discovery.inductive.util import tree_consistency
from pm4py.algo.discovery.inductive.variants.im_clean.log_im import (
    __is_base_case_act,
    __is_base_case_silent,
)
from pm4py.objects.log.util import filtering_utils
from pm4py.util import xes_constants
from pm4py import get_event_attribute_values, filter_log
from pm4py.algo.discovery.dfg import algorithm as discover_dfg
from pm4py.algo.discovery.inductive.variants.im_clean.cuts import (
    sequence as sequence_cut,
    xor as xor_cut,
    concurrency as concurrent_cut,
    loop as loop_cut,
)
from pm4py.algo.discovery.inductive.variants.im_clean.fall_throughs import (
    activity_once_per_trace,
    activity_concurrent,
    strict_tau_loop,
    tau_loop,
)
from pm4py.algo.discovery.inductive.variants.im_clean.utils import (
    __filter_dfg_on_threshold,
    __flower,
)
from pm4py.algo.discovery.minimum_self_distance import algorithm as msd_algo
from pm4py.algo.discovery.minimum_self_distance import utils as msdw_algo
from pm4py.objects.dfg.utils import dfg_utils
from pm4py.objects.log.obj import EventLog
from pm4py.statistics.end_activities.log import get as get_ends
from pm4py.statistics.start_activities.log import get as get_starters
from pm4py.util import constants

from distributed_discovery.objects.log import DistributedEventLog
from distributed_discovery.objects.process_tree import ProcessTree
from distributed_discovery.discovery.message_flow import discover_message_flows
from distributed_discovery.util.filter import (
    get_participants_event_logs,
    filter_out_dropped_messages,
)
from distributed_discovery.objects.process_tree_operator import Operator
from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.util.process_tree import fold, tree_sort


def discover_sent_received_messages_per_participant(
    log: DistributedEventLog,
) -> Tuple[
    Dict[str, Dict[str, Set[MessageFlow]]], Dict[str, Dict[str, Set[MessageFlow]]]
]:
    """
    Discovers message flows in a distributed event log.

    Parameters
    ----------
    log
        The distributed event log.

    Returns
    -------
    sent_messages_per_participant
        The discovered sending message activities per participant with their target.
    received_messages_per_participant
        The discovered receiving message activities per participant with their origin.

    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    exchanged_messages = discover_message_flows(log)
    exchanged_messages = filter_out_dropped_messages(exchanged_messages)

    sent_messages_per_participant = {
        participant: {} for participant in log.participants
    }

    received_messages_per_participant = {
        participant: {} for participant in log.participants
    }

    for sent, received in exchanged_messages:
        sent_messages = sent_messages_per_participant[sent.participant]

        # sent activities
        if sent.activity in sent_messages:
            sent_messages[sent.activity].add(received)
        else:
            sent_messages[sent.activity] = {received}

        # received activities
        received_messages = received_messages_per_participant[received.participant]
        if received.activity in received_messages:
            received_messages[received.activity].add(sent)
        else:
            received_messages[received.activity] = {sent}

    return sent_messages_per_participant, received_messages_per_participant


def discover_process_tree(
    log: DistributedEventLog, noise_threshold: float = 0.0
) -> Tuple[ProcessTree, Dict[str, Dict[str, Set[MessageFlow]]]]:
    """
    Discovers a single process tree of a distributed event log including its message flows.

    Parameters
    ----------
    log
        The distributed event log.
    noise_threshold
        Noise threshold (default: 0.0)

    Returns
    -------
    tree
        The discovered process tree of the event log.
    sent_messages_per_participant
        A dictionary containing all sending activities with their targets.
    """
    if not isinstance(log, DistributedEventLog):
        raise TypeError("the method can be applied only to a 'DistributedEventLog!")

    (
        sent_messages_per_participant,
        received_messages_per_participant,
    ) = discover_sent_received_messages_per_participant(log)

    tree = ProcessTree(Operator.PARTICIPANT, None)

    for participant, event_log in get_participants_event_logs(log).items():
        sent_messages = sent_messages_per_participant[participant]
        received_messages = received_messages_per_participant[participant]
        sub_tree = apply_tree(
            event_log, sent_messages, received_messages, noise_threshold
        )
        sub_tree.label = participant
        tree.children.append(sub_tree)

    tree_consistency.fix_parent_pointers(tree)

    return tree, sent_messages_per_participant


def apply_tree(
    event_log: EventLog,
    sent_activities: Dict[str, Set[MessageFlow]],
    received_activities: Dict[str, Set[MessageFlow]],
    threshold: float = 0.0,
) -> ProcessTree:
    """
    Applies the inductive miner on an event log and also sanitizes the process tree.

    Parameters
    ----------
    event_log
        The event log.
    sent_activities
        A dictionary containing all sending activities with their targets.
    received_activities
        A dictionary containing all receiving activities with their origin.
    threshold
        The threshold to use. Default is 0.

    Returns
    -------
    tree
        The discovered process tree of the event log.
    """
    if threshold == 0.0:
        # keep one trace per variant; more performant
        event_log = filtering_utils.keep_one_trace_per_variant(event_log)

    tree = __inductive_miner(
        event_log,
        discover_dfg.apply(event_log),
        sent_activities,
        received_activities,
        threshold,
        None,
        xes_constants.DEFAULT_NAME_KEY,
        True,
    )

    tree_consistency.fix_parent_pointers(tree)
    tree = fold(tree)
    tree_sort(tree)

    return tree


# flake8: noqa: C901
def __inductive_miner(
    log: EventLog,
    dfg: Dict[Tuple[str, str], float],
    sent_activities: Dict[str, Set[MessageFlow]],
    received_activities: Dict[str, Set[MessageFlow]],
    threshold: float,
    root: Optional[ProcessTree],
    act_key: str,
    use_msd: bool,
    remove_noise=False,
) -> ProcessTree:
    """
    Applies the inductive miner discovery algorithm on an event log.

    Parameters
    ----------
    log
        The event log.
    dfg
        The directly-follows graph of the event log.
    sent_activities
        A dictionary containing all sending activities with their targets.
    received_activities
        A dictionary containing all receiving activities with their origin.
    threshold
        The threshold to use.
    act_key
        Key used for activities in the event log.
    use_msd
        Signals if to use minimum self distance witnesses.
    remove_noise
        Signals if to remove noise from the event log.

    Returns
    -------
    pt
        The discovered process tree of the event log.
    """
    alphabet = get_event_attribute_values(log, act_key)
    if threshold > 0 and remove_noise:
        end_activities = get_ends.get_end_activities(
            log, parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key}
        )

        dfg = __filter_dfg_on_threshold(dfg, end_activities, threshold)

    original_length = len(log)
    log = filter_log(lambda t: len(t) > 0, log)

    # revised EMPTYSTRACES
    if original_length - len(log) > original_length * threshold:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.XOR, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            [EventLog(), log],
            use_msd,
        )

    start_activities = get_starters.get_start_activities(
        log, parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key}
    )
    end_activities = get_ends.get_end_activities(
        log, parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key}
    )

    # Base Case
    if __is_base_case_act(log, act_key):

        if is_message_activity(log, act_key, sent_activities):
            return add_message_activity(
                Operator.SENT, sent_activities, log[0][0][act_key], root
            )

        if is_message_activity(log, act_key, received_activities):
            return add_message_activity(
                Operator.RECEIVED,
                received_activities,
                log[0][0][act_key],
                root,
            )

    if __is_base_case_act(log, act_key) or __is_base_case_silent(log):
        return __apply_base_case(log, root, act_key)

    pre, post = dfg_utils.get_transitive_relations(dfg, alphabet)

    # Sequence
    cut = sequence_cut.detect(alphabet, pre, post)
    if cut is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.SEQUENCE, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            sequence_cut.project(log, cut, act_key),
            use_msd,
        )
    # XOR
    cut = xor_cut.detect(dfg, alphabet)
    if cut is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.XOR, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            xor_cut.project(log, cut, act_key),
            use_msd,
        )
    # Concurrent
    cut = concurrent_cut.detect(
        dfg,
        alphabet,
        start_activities,
        end_activities,
        msd=msdw_algo.derive_msd_witnesses(
            log,
            msd_algo.apply(
                log, parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key}
            ),
            parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key},
        )
        if use_msd
        else None,
    )
    if cut is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.PARALLEL, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            concurrent_cut.project(log, cut, act_key),
            use_msd,
        )
    # LOOP
    cut = loop_cut.detect(dfg, alphabet, start_activities, end_activities)
    if cut is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.LOOP, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            loop_cut.project(log, cut, act_key),
            use_msd,
        )

    aopt = activity_once_per_trace.detect(log, alphabet, act_key)
    if aopt is not None:
        operator = ProcessTree(operator=Operator.PARALLEL, parent=root)
        operator.children.append(
            __add_aopt_child_operator(
                aopt, operator, sent_activities, received_activities
            )
        )
        return __add_operator_recursive_logs(
            operator,
            sent_activities,
            received_activities,
            threshold,
            act_key,
            activity_once_per_trace.project(log, aopt, act_key),
            use_msd,
        )
    act_conc = activity_concurrent.detect(log, alphabet, act_key, use_msd)
    if act_conc is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.PARALLEL, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            activity_concurrent.project(log, act_conc, act_key),
            use_msd,
        )
    stl = strict_tau_loop.detect(log, start_activities, end_activities, act_key)
    if stl is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.LOOP, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            [stl, EventLog()],
            use_msd,
        )
    tl = tau_loop.detect(log, start_activities, act_key)
    if tl is not None:
        return __add_operator_recursive_logs(
            ProcessTree(Operator.LOOP, root),
            sent_activities,
            received_activities,
            threshold,
            act_key,
            [tl, EventLog()],
            use_msd,
        )

    if threshold > 0 and not remove_noise:
        return __inductive_miner(
            log,
            dfg,
            sent_activities,
            received_activities,
            threshold,
            root,
            act_key,
            use_msd,
            remove_noise=True,
        )

    return __flower(alphabet, root)


def __add_operator_recursive_logs(
    operator: ProcessTree,
    sent_activities: Dict[str, Set[MessageFlow]],
    received_activities: Dict[str, Set[MessageFlow]],
    threshold: float,
    act_key: str,
    logs: List[EventLog],
    use_msd: bool,
) -> ProcessTree:
    """
    Adds an operator to a process tree.

    Parameters
    ----------
    operator
        The process tree.
    sent_activities
        A dictionary containing all sending activities with their targets.
    received_activities
        A dictionary containing all receiving activities with their origin.
    threshold
        The threshold to use.
    act_key
        Key used for activities in the event log.
    logs
        The recursive log.
    use_msd
        Signals if to use minimum self distance witnesses.

    Returns
    -------
    tree
        The process tree including child nodes.
    """
    if operator.operator != Operator.LOOP:
        for log in logs:
            operator.children.append(
                __inductive_miner(
                    log,
                    discover_dfg.apply(
                        log,
                        parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key},
                    ),
                    sent_activities,
                    received_activities,
                    threshold,
                    operator,
                    act_key,
                    use_msd,
                )
            )
    else:
        operator.children.append(
            __inductive_miner(
                logs[0],
                discover_dfg.apply(
                    logs[0],
                    parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key},
                ),
                sent_activities,
                received_activities,
                threshold,
                operator,
                act_key,
                use_msd,
            )
        )
        logs = logs[1:]
        if len(logs) == 1:
            operator.children.append(
                __inductive_miner(
                    logs[0],
                    discover_dfg.apply(
                        logs[0],
                        parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: act_key},
                    ),
                    sent_activities,
                    received_activities,
                    threshold,
                    operator,
                    act_key,
                    use_msd,
                )
            )
        else:
            operator.children.append(
                __add_operator_recursive_logs(
                    ProcessTree(operator=Operator.XOR, parent=operator),
                    sent_activities,
                    received_activities,
                    threshold,
                    act_key,
                    logs,
                    use_msd,
                )
            )
    return operator


def __add_aopt_child_operator(
    activity: str,
    root: Optional[ProcessTree],
    sent_activities: Dict[str, Set[MessageFlow]],
    received_activities: Dict[str, Set[MessageFlow]],
) -> ProcessTree:
    """
    Creates the correct child operator for `activity once per trace` case in inductive miner.

    Parameters
    ----------
    activity
        The activity name.
    root
        The root process tree.
    sent_activities
        A dictionary containing all sending activities with their targets.
    received_activities
        A dictionary containing all receiving activities with their origin.

    Returns
    -------
    pt
        The newly created process tree.
    """
    if activity in sent_activities:
        return add_message_activity(Operator.SENT, sent_activities, activity, root)

    if activity in received_activities:
        return add_message_activity(
            Operator.RECEIVED, received_activities, activity, root
        )

    return ProcessTree(operator=None, parent=root, label=activity)


def add_message_activity(
    operator: Operator,
    message_dict: Dict[str, Set[MessageFlow]],
    activity: str,
    root: Optional[ProcessTree],
) -> ProcessTree:
    """
    Adds either a sent or received process tree to the current process tree including every target.

    Parameters
    ----------
    operator
        The operator that should be added either `Operator.SENT` or `Operator.RECEIVED`.
    message_dict
        A dictionary containing all target activities of this message flow.
    activity
        The activity name.
    root
        The root process tree.

    Returns
    -------
    message_activity
        The newly created process tree.

    """
    message_activity = ProcessTree(operator, root)

    node = ProcessTree(parent=message_activity, label=activity)
    message_activity.children.append(node)

    for target in message_dict[activity]:
        target = ProcessTree(
            parent=message_activity,
            label=target.activity,
            properties={"participant": target.participant},
        )
        message_activity.children.append(target)

    return message_activity


def is_message_activity(
    log: EventLog, act_key: str, messages: Dict[str, Set[MessageFlow]]
) -> bool:
    """
    Checks if activity is included in the `messages` dictionary.

    Parameters
    ----------
    log
        The log of the activity.
    act_key
        Key used for activities in the event log.
    messages
        A dictionary of sent or received messages

    Returns
    -------
    bool
        True if activity is part of dictionary

    """
    return log[0][0][act_key] in messages


def __apply_base_case(
    log: EventLog, root: Optional[ProcessTree], act_key: str
) -> ProcessTree:
    """
    Handles the base case for a process tree.
    Adds either a normal activity leaf or an empty.

    Parameters
    ----------
    log
        The log of the activity.
    root
        The root process tree.
    act_key
        Key used for activities in the event log.

    Returns
    -------
    pt
        The newly created process tree leaf.

    """
    if len(log) == 0:
        return ProcessTree(parent=root)

    return ProcessTree(parent=root, label=log[0][0][act_key])
