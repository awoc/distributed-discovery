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
import copy

from pm4py.objects.log.obj import EventLog, Trace


class DistributedTrace(Trace):
    """
    Extends the class of PM4Py 'Trace' to include the 'participant' attribute
    that reference to which participant the trace belongs to.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a 'DistributedTrace'.
        Attribute 'participant' is set after initialization.

        Parameters
        ----------
        args
        kwargs
        """
        super().__init__(*args, **kwargs)
        self.participant = None

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.participant == other.participant

    def __repr__(self, ret_list=False):
        ret = {"participant": self.participant, "attributes": self._attributes}

        if len(self._list) == 0:
            ret["events"] = []
        elif len(self._list) == 1:
            ret["events"] = [self._list[0]]
        else:
            ret["events"] = [self._list[0], "..", self._list[-1]]

        if ret_list:
            return ret

        return str(ret)

    def __copy__(self):
        new_attributes = {}
        for k, v in self.attributes.items():
            new_attributes[k] = v

        trace = DistributedTrace(attributes=new_attributes)

        for ev in self._list:
            trace.append(ev)

        trace.participant = self.participant

        return trace

    def __deepcopy__(self, memodict=None):
        new_attributes = {}
        for k, v in self.attributes.items():
            if isinstance(new_attributes, dict):
                new_attributes[k] = copy.deepcopy(v)
            else:
                new_attributes[k] = v
        trace = DistributedTrace(attributes=new_attributes)
        for ev in self._list:
            trace.append(copy.deepcopy(ev))

        trace.participant = self.participant

        return trace


class DistributedEventLog(EventLog):
    """
    Extends the class of PM4Py 'Eventlog' to include the 'participants' attribute,
    which is a set of all participants existing in an event log.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.participants = set()

    def __eq__(self, other):
        return super().__eq__(other) and self.participants == other.participants

    def __repr__(self):
        return f"participants: {self.participants}, traces: {super().__repr__()}"

    def __copy__(self):
        log = DistributedEventLog()
        log.participants = copy.copy(self.participants)
        log._attributes = copy.copy(self._attributes)
        log._extensions = copy.copy(self._extensions)
        log._omni = copy.copy(self._omni)
        log._classifiers = copy.copy(self._classifiers)
        log._properties = copy.copy(self._properties)
        for trace in self._list:
            log._list.append(trace)
        return log

    def __deepcopy__(self, memodict=None):
        log = DistributedEventLog()
        log.participants = copy.deepcopy(self.participants)
        log._attributes = copy.deepcopy(self._attributes)
        log._extensions = copy.deepcopy(self._extensions)
        log._omni = copy.deepcopy(self._omni)
        log._classifiers = copy.deepcopy(self._classifiers)
        log._properties = copy.deepcopy(self._properties)
        for trace in self._list:
            log._list.append(copy.deepcopy(trace))
        return log
