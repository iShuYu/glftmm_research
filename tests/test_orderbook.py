import unittest

from core.manager import Manager, SymbolRules
from core.orderbook import OrderBook


class OrderBookFastPathTest(unittest.TestCase):
    def test_replace_single_replaces_book_without_merge(self):
        book = OrderBook(increasing=True)
        book.merge([(20, 1), (10, 1)])

        book.replace_single((15, 3))

        self.assertEqual(book.snapshot(), [(15, 3)])

    def test_single_level_delete_head_matches_and_updates_level(self):
        book = OrderBook(increasing=True, mode=OrderBook.STRICT)
        book.replace_single((100, 5))

        self.assertEqual(book.delete_head(100, 10), [])
        self.assertEqual(book.delete_head(101, 2), [(100, 2)])
        self.assertEqual(book.snapshot(), [(100, 3)])
        self.assertEqual(book.delete_head(101, 10), [(100, 3)])
        self.assertEqual(book.snapshot(), [])


class ManagerSingleLevelPlacementTest(unittest.TestCase):
    def test_maker_single_levels_replace_existing_multilevel_books(self):
        manager = Manager(symbol_rules=SymbolRules())
        manager.place_maker_levels(
            ask_levels=[(101.0, 1.0), (102.0, 1.0)],
            bid_levels=[(98.0, 1.0), (97.0, 1.0)],
            best_ask=100.0,
            best_bid=99.0,
        )

        manager.place_maker_single_levels(
            ask_levels=[(103.0, 2.0)],
            bid_levels=[],
            best_ask=100.0,
            best_bid=99.0,
        )

        self.assertEqual(manager.books.ask_maker.snapshot(), [(103, 2)])
        self.assertEqual(manager.books.bid_maker.snapshot(), [])

    def test_maker_single_levels_reject_multiple_levels(self):
        manager = Manager(symbol_rules=SymbolRules())

        with self.assertRaisesRegex(ValueError, "at most one level per side"):
            manager.place_maker_single_levels(
                ask_levels=[(101.0, 1.0), (102.0, 1.0)],
                bid_levels=[],
                best_ask=100.0,
                best_bid=99.0,
            )

    def test_maker_steps_side_replaces_only_selected_side(self):
        manager = Manager(symbol_rules=SymbolRules())
        manager.place_maker_steps(
            ask_price_ticks=105,
            ask_qty_steps=2,
            bid_price_ticks=95,
            bid_qty_steps=3,
            best_ask_ticks=101,
            best_bid_ticks=99,
        )

        manager.place_maker_steps_side(
            side="buy",
            price_ticks=94,
            qty_steps=4,
            best_ask_ticks=101,
            best_bid_ticks=99,
        )

        self.assertEqual(manager.books.ask_maker.snapshot(), [(105, 2)])
        self.assertEqual(manager.books.bid_maker.snapshot(), [(94, 4)])

        manager.place_maker_steps_side(
            side="sell",
            price_ticks=None,
            qty_steps=0,
            best_ask_ticks=101,
            best_bid_ticks=99,
        )

        self.assertEqual(manager.books.ask_maker.snapshot(), [])
        self.assertEqual(manager.books.bid_maker.snapshot(), [(94, 4)])


if __name__ == "__main__":
    unittest.main()
