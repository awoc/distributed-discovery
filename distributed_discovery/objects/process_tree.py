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
from enum import Enum

from distributed_discovery.objects.process_tree_operator import Operator


class ProcessTree:
    """
    Process tree object adapted from PM4Py to support additional operators.
    """

    class OperatorState(Enum):
        """
        State of the operator
        """

        ENABLED = "enabled"
        OPEN = "open"
        CLOSED = "closed"
        FUTURE = "future"

    def __init__(
        self, operator=None, parent=None, children=None, label=None, properties=None
    ):
        """
        Constructor

        Parameters
        ------------
        operator
            Operator (of the current node) of the process tree
        parent
            Parent node (of the current node)
        children
            List of children of the current node
        label
            Label (of the current node)
        """
        self._operator = operator
        self._parent = parent
        self._children = [] if children is None else children
        self._label = label
        self._properties = {} if properties is None else properties

    # flake8: noqa: C901
    def __hash__(self):
        if self.label is not None:
            return hash(self.label)
        if len(self.children) == 0:
            return 37

        h = 1337
        for i in range(len(self.children)):
            h += 41 * i * hash(self.children[i])
        if self.operator == Operator.SEQUENCE:
            h = h * 13
        elif self.operator == Operator.XOR:
            h = h * 17
        elif self.operator == Operator.OR:
            h = h * 23
        elif self.operator == Operator.PARALLEL:
            h = h * 29
        elif self.operator == Operator.LOOP:
            h = h * 37
        elif self.operator == Operator.INTERLEAVING:
            h = h * 41
        elif self.operator == Operator.SENT:
            h = h * 49
        elif self.operator == Operator.RECEIVED:
            h = h * 58
        elif self.operator == Operator.PARTICIPANT:
            h = h * 67
        return h % 268435456

    def _set_operator(self, operator):
        self._operator = operator

    def _set_parent(self, parent):
        self._parent = parent

    def _set_label(self, label):
        self._label = label

    def _set_children(self, children):
        self._children = children

    def _set_properties(self, properties):
        self._properties = properties

    def _get_children(self):
        return self._children

    def _get_parent(self):
        return self._parent

    def _get_operator(self):
        return self._operator

    def _get_label(self):
        return self._label

    def _get_properties(self):
        return self._properties

    def __eq__(self, other):
        if isinstance(other, ProcessTree):
            if self.label is not None:
                return True if other.label == self.label else False
            if len(self.children) == 0:
                return other.label is None and len(other.children) == 0

            if self.operator == other.operator:
                if len(self.children) != len(other.children):
                    return False

                for i in range(len(self.children)):
                    if self.children[i] != other.children[i]:
                        return False
                return True

            return False
        return False

    def __repr__(self):
        """
        Returns a string representation of the process tree

        Returns
        ------------
        stri
            String representation of the process tree
        """
        if self.operator is not None:
            rep = str(self._operator) + "( "
            for i in range(0, len(self._children)):
                child = self._children[i]
                if len(child.children) == 0:
                    if child.label is not None:
                        rep += (
                            "'" + str(child) + "'" + ", "
                            if i < len(self._children) - 1
                            else "'" + str(child) + "'"
                        )
                    else:
                        rep += (
                            str(child) + ", "
                            if i < len(self._children) - 1
                            else str(child)
                        )
                else:
                    rep += (
                        str(child) + ", " if i < len(self._children) - 1 else str(child)
                    )
            return rep + " )"
        if self.label is not None:
            return self.label

        return "tau"

    def __str__(self):
        """
        Returns a string representation of the process tree

        Returns
        ------------
        stri
            String representation of the process tree
        """
        return self.__repr__()

    parent = property(_get_parent, _set_parent)
    children = property(_get_children, _set_children)
    operator = property(_get_operator, _set_operator)
    label = property(_get_label, _set_label)
    properties = property(_get_properties, _set_properties)
