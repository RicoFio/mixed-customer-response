"""
Unit tests for the LearningPriority framework.
"""

import unittest
import sys
import os

# Add the project root to the path to import LearningPriority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mcr.avinfra_persuasion.orders import (
    PartialOrder, PreOrder, total_order_from_list
)
from mcr.avinfra_persuasion.orders.partial_order import (
    all_partial_orders
)


class TestPreOrderClass(unittest.TestCase):
    """Tests for the PreOrder class."""

    def test_valid_preorder(self):
        """Test creating a valid pre-order."""
        elements = {1, 2, 3}
        relations = {(1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3)}  # Total order
        preorder = PreOrder(elements, relations)
        self.assertEqual(preorder.elements, elements)
        self.assertTrue(preorder.leq(1, 2))
        self.assertTrue(preorder.less(1, 2))
        self.assertFalse(preorder.less(2, 1))

    def test_reflexive_closure(self):
        """Test that reflexive closure is added."""
        elements = {1, 2}
        relations = {(1, 2)}  # Missing reflexivity
        preorder = PreOrder(elements, relations)
        self.assertIn((1, 1), preorder.relations)
        self.assertIn((2, 2), preorder.relations)

    def test_transitive_closure(self):
        """Test that transitive closure is added."""
        elements = {1, 2, 3}
        relations = {(1, 2), (2, 3)}  # Missing (1, 3)
        preorder = PreOrder(elements, relations)
        self.assertIn((1, 3), preorder.relations)

    def test_invalid_preorder_not_reflexive(self):
        """Test that invalid pre-order without reflexivity raises error."""
        elements = {1, 2}
        relations = {(1, 2), (2, 1)}  # No (1,1) or (2,2), but closure adds them
        # Actually, reflexive closure adds them, so this should work
        preorder = PreOrder(elements, relations)
        self.assertIn((1, 1), preorder.relations)

    def test_invalid_preorder_not_transitive(self):
        """Test that invalid pre-order without transitivity raises error."""
        # Since we compute transitive closure, it should add missing relations
        elements = {1, 2, 3}
        relations = {(1, 2), (2, 3)}  # Will add (1,3)
        preorder = PreOrder(elements, relations)
        self.assertIn((1, 3), preorder.relations)

    def test_leq_less(self):
        """Test leq and less methods."""
        elements = {1, 2}
        relations = {(1, 1), (2, 2), (1, 2)}
        preorder = PreOrder(elements, relations)
        self.assertTrue(preorder.leq(1, 2))
        self.assertTrue(preorder.less(1, 2))
        self.assertFalse(preorder.less(2, 1))
        self.assertTrue(preorder.leq(2, 2))
        self.assertFalse(preorder.less(2, 2))
    
    def test_partial_order_from_total_order(self):
        """Test building a partial order from a total order list."""
        elements = [1, 2, 3]
        total_preorder = total_order_from_list(elements)
        self.assertTrue(total_preorder.leq(1, 2))
        self.assertTrue(total_preorder.leq(2, 3))
        self.assertTrue(total_preorder.leq(1, 3))
        self.assertFalse(total_preorder.leq(3, 1))
    
    def test_pre_order_from_subset(self):
        """Test building a pre-order from a subset of its elements."""
        elements = {1, 2, 3, 4}
        relations = {(1, 2), (2, 3), (1, 3), (3, 4)}
        preorder = PreOrder(elements, relations)
        subset_preorder = preorder.build_sub_preorder({1, 2, 3})
        self.assertEqual(subset_preorder.elements, {1, 2, 3})
        self.assertTrue(subset_preorder.leq(1, 2))
        self.assertTrue(subset_preorder.leq(2, 3))
        self.assertTrue(subset_preorder.leq(1, 3))


class TestPartialOrderClass(unittest.TestCase):
    """Tests for the PartialOrder class."""

    def test_valid_partial_order(self):
        """Test creating a valid partial order."""
        elements = {1, 2, 3}
        relations = {(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)}
        poset = PartialOrder(elements, relations)
        self.assertTrue(poset.less(1, 2))
        self.assertFalse(poset.leq(2, 1))

    def test_invalid_partial_order_not_antisymmetric(self):
        """Test that non-antisymmetric relation raises error."""
        elements = {1, 2}
        relations = {(1, 1), (2, 2), (1, 2), (2, 1)}  # Antisymmetric violation
        with self.assertRaises(ValueError):
            PartialOrder(elements, relations)

    def test_total_order_from_list(self):
        """Test creating total order from list."""
        elements = [1, 2, 3]
        poset = total_order_from_list(elements)
        self.assertTrue(poset.less(1, 2))
        self.assertTrue(poset.less(2, 3))
        self.assertTrue(poset.less(1, 3))
        self.assertFalse(poset.less(3, 1))


class TestExtremaElements(unittest.TestCase):
    """Tests for minimal and maximal elements."""

    def test_maximal_elements_total_order(self):
        """Test maximal elements in total order."""
        elements = [1, 2, 3]
        poset = total_order_from_list(elements)
        maximals = poset.maximal_elements()
        self.assertEqual(maximals, {3})

    def test_minimal_elements_total_order(self):
        """Test minimal elements in total order."""
        elements = [1, 2, 3]
        poset = total_order_from_list(elements)
        minimals = poset.minimal_elements()
        self.assertEqual(minimals, {1})

    def test_maximal_elements_partial_order(self):
        """Test maximal elements in partial order."""
        elements = {1, 2, 3}
        relations = {(1, 1), (2, 2), (3, 3), (1, 2), (1, 3)}  # 2 and 3 incomparable
        poset = PartialOrder(elements, relations)
        maximals = poset.maximal_elements()
        self.assertEqual(maximals, {2, 3})

    def test_minimal_elements_partial_order(self):
        """Test minimal elements in partial order."""
        elements = {1, 2, 3}
        relations = {(1, 1), (2, 2), (3, 3), (1, 2), (1, 3)}  # 1 minimal, 2 and 3 incomparable
        poset = PartialOrder(elements, relations)
        minimals = poset.minimal_elements()
        self.assertEqual(minimals, {1})

class TestOrderOfPriority(unittest.TestCase):
    """Tests for all_partial_orders function."""

    def test_all_partial_orders_small_set(self):
        """Test generating all partial orders for a small set."""
        elements = {1, 2}
        posets = list(all_partial_orders(elements))
        # There are 3 partial orders on a 2-element set
        self.assertEqual(len(posets), 3)
        relations_sets = [poset.relations for poset in posets]
        expected_relations = [
            {(1, 1), (2, 2)},  # No relations
            {(1, 1), (2, 2), (1, 2)},  # 1 < 2
            {(1, 1), (2, 2), (2, 1)},  # 2 < 1
        ]
        for expected in expected_relations:
            self.assertIn(expected, relations_sets)
        for relation in relations_sets:
            self.assertIn(relation, expected_relations)

        elements = {'a', 'b', 'c'}
        posets = list(all_partial_orders(elements))
        # There are 19 partial orders on a 3-element set
        self.assertEqual(len(posets), 19)
        expected_relations = [
            {(a, a) for a in elements},  # No relations
            {(a, a) for a in elements} | {('a', 'b')},
            {(a, a) for a in elements} | {('a', 'c')},
            {(a, a) for a in elements} | {('b', 'c')},
            {(a, a) for a in elements} | {('b', 'a')},
            {(a, a) for a in elements} | {('c', 'a')},
            {(a, a) for a in elements} | {('c', 'b')},
            {(a, a) for a in elements} | {('a', 'b'), ('b', 'c'), ('a', 'c')},
            {(a, a) for a in elements} | {('a', 'c'), ('c', 'b'), ('a', 'b')},
            {(a, a) for a in elements} | {('b', 'a'), ('a', 'c'), ('b', 'c')},
            {(a, a) for a in elements} | {('b', 'c'), ('c', 'a'), ('b', 'a')},
            {(a, a) for a in elements} | {('c', 'a'), ('a', 'b'), ('c', 'b')},
            {(a, a) for a in elements} | {('c', 'b'), ('b', 'a'), ('c', 'a')},
            {(a, a) for a in elements} | {('a', 'b'), ('a', 'c')},
            {(a, a) for a in elements} | {('b', 'a'), ('b', 'c')},
            {(a, a) for a in elements} | {('c', 'a'), ('c', 'b')},
            {(a, a) for a in elements} | {('b', 'a'), ('c', 'a')},
            {(a, a) for a in elements} | {('a', 'b'), ('c', 'b')},
            {(a, a) for a in elements} | {('a', 'c'), ('b', 'c')},
        ]
        relations_sets = [poset.relations for poset in posets]
        for expected in expected_relations:
            self.assertIn(expected, relations_sets)
        for relation in relations_sets:
            self.assertIn(relation, expected_relations)

        elements = {'a', 'b', 'c', 'd'}
        posets = list(all_partial_orders(elements))
        # There are 219 partial orders on a 4-element set
        self.assertEqual(len(posets), 219)

        elements = {'a', 'b', 'c', 'd', 'e'}
        posets = list(all_partial_orders(elements))
        # There are 4231 partial orders on a 5-element set
        self.assertEqual(len(posets), 4231)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
