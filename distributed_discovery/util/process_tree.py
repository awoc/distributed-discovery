"""
    This file contains modified code from PM4Py
    and adds operators for process trees to handle distributed processes.

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
import hashlib

from pm4py.objects.process_tree.utils.generic import is_tau_leaf
from pm4py.util import constants

from distributed_discovery.objects.process_tree_operator import Operator


def fold(tree):
    """
    This method reduces a process tree by merging nodes of the form N(N(a,b),c) into N(a,b,c), i.e., where
    N = || or X. For example X(X(a,b),c) == X(a,b,c).
    Furthermore, meaningless parts, e.g., internal nodes without children, or, operators with one child are removed
    as well.

    :param tree:
    :return:
    """
    tree = copy.deepcopy(tree)
    tree = _fold(tree)
    root = tree
    while root.parent is None and len(tree.children) == 1:
        root = tree.children[0]
        root.parent = None
        tree.children.clear()
        del tree
        tree = root
    if str(reduce_tau_leafs(copy.deepcopy(tree))) != str(tree):
        tree = fold(tree)
    return tree


def _fold(tree):
    tree = reduce_tau_leafs(tree)
    if len(tree.children) > 0:
        tree.children = list(map(lambda c: _fold(c), tree.children))
        tree.children = list(filter(lambda c: c is not None, tree.children))
        if len(tree.children) == 0:
            tree.parent = None
            tree.children = None
            return None
        if len(tree.children) == 1:
            child = tree.children[0]
            child.parent = tree.parent
            tree.parent = None
            tree.children = None
            return child
        if tree.operator in [
            Operator.SEQUENCE,
            Operator.XOR,
            Operator.PARALLEL,
        ]:
            children = copy.deepcopy(tree.children)
            for c in children:
                if c.operator == tree.operator:
                    i = tree.children.index(c)
                    tree.children[i:i] = c.children
                    for cc in c.children:
                        cc.parent = tree
                    tree.children.remove(c)
                    c.children.clear()
                    c.parent = None
    return tree


# flake8: noqa: C901
def reduce_tau_leafs(tree):
    """
    This method reduces tau leaves that are not meaningful. For example tree ->(a,\tau,b) is reduced to ->(a,b).
    In some cases this results in constructs such as ->(a), i.e., a sequence with a single child. Such constructs
    are not further reduced.

    :param tree:
    :return:
    """
    if len(tree.children) > 0:
        for c in tree.children:
            reduce_tau_leafs(c)
        silents = 0
        for c in tree.children:
            if is_tau_leaf(c):
                silents += 1
        if silents > 0:
            if len(tree.children) == silents:
                # all children are tau, keep one (might be folded later)
                if tree.operator in [
                    Operator.SEQUENCE,
                    Operator.PARALLEL,
                    Operator.XOR,
                    Operator.OR,
                ]:
                    # remove all but one, later reductions might need the fact that skipping is possible
                    while silents > 1:
                        cc = tree.children
                        for c in cc:
                            if is_tau_leaf(c):
                                c.parent = None
                                tree.children.remove(c)
                                silents -= 1
                                break
                elif tree.operator == Operator.LOOP and len(tree.children) == 2:
                    # remove all loop is redundant
                    cc = tree.children
                    for c in cc:
                        if is_tau_leaf(c):
                            c.parent = None
                            tree.children.remove(c)
            else:
                # at least one non-tau child
                if tree.operator in [Operator.SEQUENCE, Operator.PARALLEL]:
                    # remove all, they are redundant for these operators
                    cc = tree.children
                    for c in cc:
                        if is_tau_leaf(c):
                            c.parent = None
                            tree.children.remove(c)
                elif tree.operator in [Operator.XOR, Operator.OR]:
                    # keep one, we should be able to skip
                    while silents > 1:
                        cc = tree.children
                        for c in cc:
                            if is_tau_leaf(c):
                                c.parent = None
                                tree.children.remove(c)
                                silents -= 1
                                break
    return tree


def tree_sort(tree):
    """
    Sort a tree in such way that the order of the nodes
    in AND/XOR children is always the same.
    This is a recursive function

    Parameters
    --------------
    tree
        Process tree
    """
    tree.labels_hash_sum = 0
    for child in tree.children:
        tree_sort(child)
        tree.labels_hash_sum += child.labels_hash_sum
    if tree.label is not None:
        # this assures that among different executions, the same string gets always the same hash
        this_hash = int(
            hashlib.md5(tree.label.encode(constants.DEFAULT_ENCODING)).hexdigest(), 16
        )
        tree.labels_hash_sum += this_hash
    if tree.operator is Operator.PARALLEL or tree.operator is Operator.XOR:
        tree.children = sorted(tree.children, key=lambda x: x.labels_hash_sum)
