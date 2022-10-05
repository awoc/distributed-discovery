"""
    This file contains modified code from PM4Py
    to handle event logs with using the `Message` extension.

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
import logging
from lxml import etree

from pm4py.objects.log.obj import Event
from pm4py.util import constants
from pm4py.util import xes_constants
from pm4py.util.dt_parsing import parser as dt_parser

from distributed_discovery.util.constants import (
    DEFAULT_MESSAGE_SENT_KEY,
    DEFAULT_MESSAGE_RECEIVED_KEY,
    DEFAULT_TRACE_PARTICIPANT_KEY,
    DEFAULT_MESSAGE_CONTENT_KEY,
)
from distributed_discovery.objects.log import DistributedEventLog, DistributedTrace


# ITERPARSE EVENTS
_EVENT_END = "end"
_EVENT_START = "start"


def read_xes(file_path: str) -> DistributedEventLog:
    """
    Imports a XES file formatted as a distributed log with different participants and message exchanges.

    Parameters
    ----------
    file_path : str
        Absolute file path where XES file is located

    Returns
    -------
    log : DistributedLog
        a list of traces as a distributed log

    """
    with open(file_path, "rb") as file:
        context = etree.iterparse(
            file, events=[_EVENT_START, _EVENT_END], encoding="UTF-8"
        )

        return import_distributed_log(context)


# flake8: noqa: C901
def import_distributed_log(context: etree.iterparse) -> DistributedEventLog:
    """
    Import a XES log from an iterparse context and format to DistributedEventLog.

    Parameters
    ----------
    context
        Iterparse context.

    Returns
    -------
     log
          DistributedEventLog

    Raises
    ------
    SyntaxError
        For wrongly formatted files.

    References
    ----------
    This function was extended from PM4Py Core.

    """
    max_no_traces_to_import = 55555

    date_parser = dt_parser.get()

    log = None
    trace = None
    event = None

    tree = {}

    for tree_event, elem in context:
        if tree_event == _EVENT_START:  # starting to read
            parent = tree[elem.getparent()] if elem.getparent() in tree else None

            if elem.tag.endswith(xes_constants.TAG_STRING):
                if (
                    elem.get(xes_constants.KEY_KEY) == DEFAULT_TRACE_PARTICIPANT_KEY
                    and trace is not None
                    and event is None
                ):
                    trace.participant = elem.get(xes_constants.KEY_VALUE)
                elif parent is not None:
                    tree = __parse_attribute(
                        elem,
                        parent,
                        elem.get(xes_constants.KEY_KEY),
                        elem.get(xes_constants.KEY_VALUE),
                        tree,
                    )

                continue

            elif elem.tag.endswith(xes_constants.TAG_DATE):
                try:
                    dt = date_parser.apply(elem.get(xes_constants.KEY_VALUE))
                    tree = __parse_attribute(
                        elem, parent, elem.get(xes_constants.KEY_KEY), dt, tree
                    )
                except TypeError:
                    logging.error(
                        "failed to parse date: "
                        + str(elem.get(xes_constants.KEY_VALUE))
                    )
                except ValueError:
                    logging.error(
                        "failed to parse date: %", elem.get(xes_constants.KEY_VALUE)
                    )
                continue

            elif elem.tag.endswith(xes_constants.TAG_EVENT):
                if event is not None:
                    raise SyntaxError("file contains <event> in another <event> tag")
                event = Event()
                tree[elem] = event
                continue

            elif elem.tag.endswith(xes_constants.TAG_TRACE):
                if len(log) >= max_no_traces_to_import:
                    break
                if trace is not None:
                    raise SyntaxError("file contains <trace> in another <trace> tag")
                trace = DistributedTrace()
                tree[elem] = trace.attributes
                continue

            elif elem.tag.endswith(xes_constants.TAG_FLOAT):
                if parent is not None:
                    try:
                        val = float(elem.get(xes_constants.KEY_VALUE))
                        tree = __parse_attribute(
                            elem, parent, elem.get(xes_constants.KEY_KEY), val, tree
                        )
                    except ValueError:
                        logging.error(
                            "failed to parse float: "
                            + str(elem.get(xes_constants.KEY_VALUE))
                        )
                continue

            elif elem.tag.endswith(xes_constants.TAG_INT):
                if parent is not None:
                    try:
                        val = int(elem.get(xes_constants.KEY_VALUE))
                        tree = __parse_attribute(
                            elem, parent, elem.get(xes_constants.KEY_KEY), val, tree
                        )
                    except ValueError:
                        logging.error(
                            "failed to parse int: "
                            + str(elem.get(xes_constants.KEY_VALUE))
                        )
                continue

            elif elem.tag.endswith(xes_constants.TAG_BOOLEAN):
                if parent is not None:
                    try:
                        val0 = elem.get(xes_constants.KEY_VALUE)
                        val = False
                        if str(val0).lower() == "true":
                            val = True
                        tree = __parse_attribute(
                            elem, parent, elem.get(xes_constants.KEY_KEY), val, tree
                        )
                    except ValueError:
                        logging.error(
                            "failed to parse boolean: "
                            + str(elem.get(xes_constants.KEY_VALUE))
                        )
                continue

            elif elem.tag.endswith(xes_constants.TAG_LIST):
                if parent is not None:
                    # lists have no value, hence we put None as a value
                    tree = __parse_attribute(
                        elem, parent, elem.get(xes_constants.KEY_KEY), None, tree
                    )
                continue

            elif elem.tag.endswith(xes_constants.TAG_ID):
                if parent is not None:
                    tree = __parse_attribute(
                        elem,
                        parent,
                        elem.get(xes_constants.KEY_KEY),
                        elem.get(xes_constants.KEY_VALUE),
                        tree,
                    )
                continue

            elif elem.tag.endswith(xes_constants.TAG_EXTENSION):
                if log is None:
                    raise SyntaxError("extension found outside of <log> tag")
                if (
                    elem.get(xes_constants.KEY_NAME) is not None
                    and elem.get(xes_constants.KEY_PREFIX) is not None
                    and elem.get(xes_constants.KEY_URI) is not None
                ):
                    log.extensions[elem.get(xes_constants.KEY_NAME)] = {
                        xes_constants.KEY_PREFIX: elem.get(xes_constants.KEY_PREFIX),
                        xes_constants.KEY_URI: elem.get(xes_constants.KEY_URI),
                    }
                continue

            elif elem.tag.endswith(xes_constants.TAG_GLOBAL):
                if log is None:
                    raise SyntaxError("global found outside of <log> tag")
                if elem.get(xes_constants.KEY_SCOPE) is not None:
                    log.omni_present[elem.get(xes_constants.KEY_SCOPE)] = {}
                    tree[elem] = log.omni_present[elem.get(xes_constants.KEY_SCOPE)]
                continue

            elif elem.tag.endswith(xes_constants.TAG_CLASSIFIER):
                if log is None:
                    raise SyntaxError("classifier found outside of <log> tag")
                if elem.get(xes_constants.KEY_KEYS) is not None:
                    classifier_value = elem.get(xes_constants.KEY_KEYS)
                    if "'" in classifier_value:
                        log.classifiers[elem.get(xes_constants.KEY_NAME)] = [
                            x for x in classifier_value.split("'") if x.strip()
                        ]
                    else:
                        log.classifiers[
                            elem.get(xes_constants.KEY_NAME)
                        ] = classifier_value.split()
                continue

            elif elem.tag.endswith(xes_constants.TAG_LOG):
                if log is not None:
                    raise SyntaxError("file contains > 1 <log> tags")
                log = DistributedEventLog()
                tree[elem] = log.attributes
                continue

        elif tree_event == _EVENT_END:
            if elem in tree:
                del tree[elem]
            elem.clear()
            if elem.getprevious() is not None:
                try:
                    del elem.getparent()[0]
                except TypeError:
                    logging.error("Failed to parse " + elem + " getparent()[0].")

            if elem.tag.endswith(xes_constants.TAG_EVENT):
                if trace is not None:
                    trace.append(event)
                    event = None
                continue

            elif elem.tag.endswith(xes_constants.TAG_TRACE):
                log.append(trace)
                log.participants.add(trace.participant)

                trace = None
                continue

            elif elem.tag.endswith(xes_constants.TAG_LOG):
                continue

    del context

    # sets the activity key as default classifier in the log's properties
    log.properties[
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    ] = xes_constants.DEFAULT_NAME_KEY
    log.properties[
        constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY
    ] = xes_constants.DEFAULT_NAME_KEY
    # sets the default timestamp key
    log.properties[
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    ] = xes_constants.DEFAULT_TIMESTAMP_KEY
    # sets the default resource key
    log.properties[
        constants.PARAMETER_CONSTANT_RESOURCE_KEY
    ] = xes_constants.DEFAULT_RESOURCE_KEY
    # sets the default transition key
    log.properties[
        constants.PARAMETER_CONSTANT_TRANSITION_KEY
    ] = xes_constants.DEFAULT_TRANSITION_KEY
    # sets the default group key
    log.properties[
        constants.PARAMETER_CONSTANT_GROUP_KEY
    ] = xes_constants.DEFAULT_GROUP_KEY

    return log


def __parse_attribute(elem, store, key, value, tree):
    if len(elem.getchildren()) == 0:
        if isinstance(store, list):
            # changes to the store of lists: not dictionaries anymore
            # but pairs of key-values.
            if key == DEFAULT_MESSAGE_CONTENT_KEY:
                message = "".join(value.split())
                store.append(message)
            else:
                store.append((key, value))
        else:
            store[key] = value
    else:
        if elem.getchildren()[0].tag.endswith(xes_constants.TAG_VALUES):
            if key in (DEFAULT_MESSAGE_SENT_KEY, DEFAULT_MESSAGE_RECEIVED_KEY):
                store[key] = []
                tree[elem] = store[key]
                tree[elem.getchildren()[0]] = tree[elem]
            else:
                store[key] = {
                    xes_constants.KEY_VALUE: value,
                    xes_constants.KEY_CHILDREN: [],
                }
                tree[elem] = store[key][xes_constants.KEY_CHILDREN]
                tree[elem.getchildren()[0]] = tree[elem]
        else:
            store[key] = {
                xes_constants.KEY_VALUE: value,
                xes_constants.KEY_CHILDREN: {},
            }
            tree[elem] = store[key][xes_constants.KEY_CHILDREN]
    return tree
