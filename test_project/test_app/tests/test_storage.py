#!/usr/bin/python
# -*- coding: utf-8 -*-


from unittest.mock import MagicMock, patch

from autocompleter import Autocompleter, base, registry, signal_registry
from autocompleter import settings as auto_settings
from autocompleter.registry import (
    add_obj_to_autocompleter,
    remove_obj_from_autocompleter,
)

from test_app import calc_info
from test_app.autocompleters import (
    CalcAutocompleteProvider,
    FacetedStockAutocompleteProvider,
    IndicatorAliasedAutocompleteProvider,
    StockAutocompleteProvider,
)
from test_app.models import Indicator, Stock
from test_app.tests.base import AutocompleterTestCase


class StoringAndRemovingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json", "indicator_test_data_small.json"]

    def test_store_and_remove(self):
        """
        Storing and removing an item works
        """
        aapl = Stock.objects.get(symbol="AAPL")
        provider = StockAutocompleteProvider(aapl)

        provider.store()
        keys = self.redis.hkeys("djac.test.stock")
        self.assertEqual(len(keys), 1)

        provider.remove()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)

    def test_store_saves_terms(self):
        """
        Storing saves norm terms, not plain terms
        """
        aapl = Stock.objects.get(symbol="AAPL")
        provider = StockAutocompleteProvider(aapl)
        provider.store()

        terms = provider.get_terms()
        norm_terms = provider._get_norm_terms(terms)
        provider_name = provider.get_provider_name()
        key = base.TERM_MAP_BASE_NAME % (provider_name,)
        terms_stored_serialized = self.redis.hget(key, aapl.id)
        terms_stored = provider._deserialize_data(terms_stored_serialized)
        self.assertEqual(norm_terms, terms_stored)
        self.assertNotEqual(terms, terms_stored)

    def test_store_score(self):
        """
        Storing and removing a score works
        """
        aapl = Stock.objects.get(symbol="AAPL")
        provider = StockAutocompleteProvider(aapl)
        provider.store()

        keys = self.redis.hkeys("djac.test.stock.sm")
        self.assertEqual(len(keys), 1)

        provider.remove()
        keys = self.redis.hkeys("djac.test.stock.sm")
        self.assertEqual(len(keys), 0)

    def test_dict_store_and_remove(self):
        """
        Storing and removing an dictionary item works
        """
        item = calc_info.calc_dicts[0]
        provider = CalcAutocompleteProvider(item)
        provider.store()

        keys = self.redis.hkeys("djac.test.metric")
        self.assertEqual(len(keys), 1)

        provider.remove()
        keys = self.redis.keys("djac.test.metric*")
        self.assertEqual(len(keys), 0)

    def test_store_and_remove_all_basic(self):
        """
        Storing and removing items all at once works for a dictionary obj autocompleter.
        """
        autocomp = Autocompleter("stock")

        autocomp.store_all()
        keys = self.redis.hkeys("djac.test.stock")
        self.assertEqual(len(keys), 104)

        autocomp.remove_all()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)

    def test_orphan_removal(self):
        """
        test orphan removal
        """
        signal_registry.register(Indicator)

        autocomp = Autocompleter("indicator")
        autocomp.store_all()

        unemployment = Indicator.objects.get(internal_name="unemployment_rate")

        unemployment.name = "free parking"
        unemployment.save()

        self.assertTrue(autocomp.suggest("free parking")[0]["id"] == 1)
        self.assertTrue(len(autocomp.suggest("US Unemployment Rate")) == 0)

        autocomp.remove_all()
        signal_registry.unregister(Indicator)

    def test_removal_when_no_longer_passing_inclusion_test(self):
        """
        test that an item is removed once it no longer passes the .include_item() check.
        """
        signal_registry.register(Indicator)

        autocomp = Autocompleter("indicator_selective")
        autocomp.store_all()

        unemployment = Indicator.objects.get(internal_name="unemployment_rate")

        unemployment.name = "Free Parking"
        unemployment.save()

        self.assertTrue(autocomp.suggest("free parking")[0]["id"] == 1)
        self.assertTrue(len(autocomp.suggest("US Unemployment Rate")) == 0)

        unemployment.name = "US Unemployment Rate"
        unemployment.save()

        self.assertTrue(len(autocomp.suggest("free parking")) == 0)
        self.assertTrue(len(autocomp.suggest("US Unemployment Rate")) == 0)

        autocomp.remove_all()
        signal_registry.unregister(Indicator)

    def test_dict_store_and_remove_all_basic(self):
        """
        Storing and removing items all at once works for a single-model autocompleter.
        """
        autocomp = Autocompleter("metric")

        autocomp.store_all()
        keys = self.redis.hkeys("djac.test.metric")
        self.assertEqual(len(keys), 8)

        autocomp.remove_all()
        keys = self.redis.keys("djac.test.metric")
        self.assertEqual(len(keys), 0)

    def test_store_and_remove_all_basic_with_caching(self):
        """
        Storing and removing items all at once works with caching turned on
        """
        # Let's turn on caching because that will store things in Redis and we want to make
        # sure we clean them up.
        setattr(auto_settings, "CACHE_TIMEOUT", 3600)

        autocomp = Autocompleter("stock")
        autocomp.store_all()

        keys = self.redis.hkeys("djac.test.stock")
        self.assertEqual(len(keys), 104)

        autocomp = Autocompleter("stock")
        for i in range(0, 3):
            autocomp.suggest("a")
            autocomp.suggest("z")
            autocomp.exact_suggest("aapl")
            autocomp.exact_suggest("xyz")

        autocomp.remove_all()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "CACHE_TIMEOUT", 0)

    def test_dict_store_and_remove_all_basic_with_caching(self):
        """
        Storing and removing items all at once works with caching turned on on dict ac
        """
        # Let's turn on caching because that will store things in Redis and we want to make
        # sure we clean them up.
        setattr(auto_settings, "CACHE_TIMEOUT", 3600)

        autocomp = Autocompleter("metric")
        autocomp.store_all()

        keys = self.redis.hkeys("djac.test.metric")
        self.assertEqual(len(keys), 8)

        autocomp = Autocompleter("metric")
        for i in range(0, 3):
            autocomp.suggest("m")
            autocomp.suggest("e")
            autocomp.exact_suggest("PE Ratio TTM")
            autocomp.exact_suggest("Market Cap")

        autocomp.remove_all()

        keys = self.redis.keys("djac.test.metric*")
        self.assertEqual(len(keys), 0)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "CACHE_TIMEOUT", 0)

    def test_store_and_remove_all_multi(self):
        """
        Storing and removing items all at once works for a multi-model autocompleter.
        """
        autocomp = Autocompleter("mixed")

        autocomp.store_all()
        keys = self.redis.hkeys("djac.test.stock")
        self.assertEqual(len(keys), 104)
        keys = self.redis.hkeys("djac.test.ind")
        self.assertEqual(len(keys), 100)
        keys = self.redis.hkeys("djac.test.metric")
        self.assertEqual(len(keys), 8)

        autocomp.remove_all()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)
        keys = self.redis.keys("djac.test.ind*")
        self.assertEqual(len(keys), 0)
        keys = self.redis.keys("djac.test.mixed*")
        self.assertEqual(len(keys), 0)
        keys = self.redis.keys("djac.test.metric*")
        self.assertEqual(len(keys), 0)

    def test_remove_intermediate_results_exact_suggest(self):
        """
        After exact_suggest call, all intermediate result sets are removed
        """
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 2)
        autocomp = Autocompleter("stock")
        autocomp.store_all()

        autocomp.exact_suggest("aapl")
        keys = self.redis.keys("djac.results.*")
        self.assertEqual(len(keys), 0)

        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 0)

    def test_remove_intermediate_results_suggest(self):
        """
        After suggest call, all intermediate result sets are removed
        """
        autocomp = Autocompleter("stock")
        autocomp.store_all()

        autocomp.suggest("aapl")
        keys = self.redis.keys("djac.results.*")
        self.assertEqual(len(keys), 0)


class FacetedStoringAndRemovingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json"]

    def test_store_facet_data(self):
        """
        Storing saves facet data
        """
        aapl = Stock.objects.get(symbol="AAPL")
        provider = FacetedStockAutocompleteProvider(aapl)
        provider.store()

        provider_name = provider.get_provider_name()
        # the FacetedStockAutocompleteProvider get_facets is set to ['sector', 'industry']
        facet_set_name = base.FACET_SET_BASE_NAME % (
            provider_name,
            "sector",
            "Technology",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 1)

        facet_set_name = base.FACET_SET_BASE_NAME % (
            provider_name,
            "industry",
            "Consumer Electronics",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 1)

        facet_map_name = base.FACET_MAP_BASE_NAME % (provider_name,)
        facet_data = provider._deserialize_data(
            self.redis.hget(facet_map_name, aapl.id)
        )
        self.assertEqual(
            facet_data,
            [
                {"key": "sector", "value": aapl.sector},
                {"key": "industry", "value": aapl.industry},
            ],
        )

    def test_second_store_removes_old_facet_data(self):
        """
        Store removes outdated facet data and updates mapping
        """
        aapl = Stock.objects.get(symbol="AAPL")
        provider = FacetedStockAutocompleteProvider(aapl)
        provider.store()

        provider_name = provider.get_provider_name()
        # the FacetedStockAutocompleteProvider get_facets is set to ['sector']
        facet_set_name = base.FACET_SET_BASE_NAME % (
            provider_name,
            "sector",
            "Technology",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 1)

        facet_map_name = base.FACET_MAP_BASE_NAME % (provider_name,)
        facet_data = provider._deserialize_data(
            self.redis.hget(facet_map_name, aapl.id)
        )
        self.assertEqual(
            facet_data,
            [
                {"key": "sector", "value": aapl.sector},
                {"key": "industry", "value": aapl.industry},
            ],
        )

        aapl.sector = "Healthcare"
        aapl.save()
        provider.store()
        # make sure the old key was removed after the second store call
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 0)
        facet_set_name = base.FACET_SET_BASE_NAME % (
            provider_name,
            "sector",
            "Healthcare",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 1)

        facet_data = provider._deserialize_data(
            self.redis.hget(facet_map_name, aapl.id)
        )
        self.assertEqual(
            facet_data,
            [
                {"key": "sector", "value": "Healthcare"},
                {"key": "industry", "value": aapl.industry},
            ],
        )

    def test_remove_facet_data(self):
        """
        Remove takes care of deleting facet data
        """
        aapl = Stock.objects.get(symbol="AAPL")
        provider = FacetedStockAutocompleteProvider(aapl)
        provider.store()
        provider.remove()

        provider_name = provider.get_provider_name()
        # the FacetedStockAutocompleteProvider get_facets is set to ['sector']
        facet_set_name = base.FACET_SET_BASE_NAME % (
            provider_name,
            "sector",
            "Technology",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 0)

        facet_map_name = base.FACET_MAP_BASE_NAME % (provider_name,)
        keys = self.redis.hkeys(facet_map_name)
        self.assertEqual(len(keys), 0)

    def test_remove_facet_data_on_failed_inclusion(self):
        """
        Remove takes care of deleting facet data when an item fails inclusion test.
        """
        signal_registry.register(Stock)
        Autocompleter("faceted_stock")
        aapl = Stock.objects.get(symbol="AAPL")
        aapl.save()

        facet_set_name = base.FACET_SET_BASE_NAME % (
            "faceted_stock",
            "sector",
            "Technology",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 1)

        facet_map_name = base.FACET_MAP_BASE_NAME % ("faceted_stock",)
        keys = self.redis.hkeys(facet_map_name)
        self.assertEqual(len(keys), 1)

        aapl.hidden = True
        aapl.save()

        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 0)

        keys = self.redis.hkeys(facet_map_name)
        self.assertEqual(len(keys), 0)

        signal_registry.unregister(Stock)

    def test_store_all_facet_data(self):
        """
        Calling store_all stores all facet data
        """
        autocomp = Autocompleter("faceted_stock")
        autocomp.store_all()
        facet_set_name = base.FACET_SET_BASE_NAME % (
            "faceted_stock",
            "sector",
            "Technology",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 12)

        facet_map_name = base.FACET_MAP_BASE_NAME % ("faceted_stock",)
        keys = self.redis.hkeys(facet_map_name)
        self.assertEqual(len(keys), 104)

    def test_remove_all_facet_data(self):
        """
        Calling remove_all clears all facet data
        """
        autocomp = Autocompleter("faceted_stock")
        autocomp.store_all()
        facet_set_name = base.FACET_SET_BASE_NAME % (
            "faceted_stock",
            "sector",
            "Technology",
        )
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 12)
        facet_map_name = base.FACET_MAP_BASE_NAME % ("faceted_stock",)
        keys = self.redis.hkeys(facet_map_name)
        self.assertEqual(len(keys), 104)

        autocomp.remove_all()
        set_length = self.redis.zcard(facet_set_name)
        self.assertEqual(set_length, 0)
        keys = self.redis.hkeys(facet_map_name)
        self.assertEqual(len(keys), 0)


class SelectiveStoringTestCase(AutocompleterTestCase):
    fixtures = ["indicator_test_data_small.json"]

    def test_selective_add_and_remove(self):
        """
        We can exclude certain objects from the autocompleter selectively.
        """
        autocomp = Autocompleter("indicator")
        autocomp.store_all()
        matches = autocomp.suggest("us unemployment rate")
        self.assertEqual(len(matches), 1)
        autocomp.remove_all()

        autocomp = Autocompleter("indicator_selective")
        autocomp.store_all()
        matches = autocomp.suggest("us unemployment rate")
        self.assertEqual(len(matches), 0)
        autocomp.remove_all()


class SignalBasedStoringTestCase(AutocompleterTestCase):
    fixtures = ["indicator_test_data_small.json"]

    def test_signal_based_add_and_remove(self):
        """
        Turning on signals will automatically add and remove and object from the autocompleter
        """
        aapl = Stock(symbol="AAPL", name="Apple", market_cap=50)
        aapl.save()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)

        signal_registry.register(Stock)

        aapl.save()
        keys = self.redis.keys("djac.test.stock*")
        self.assertNotEqual(len(keys), 0)

        aapl.delete()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)

        signal_registry.unregister(Stock)

    @patch("autocompleter.base.AutocompleterProviderBase.store")
    @patch("autocompleter.base.AutocompleterProviderBase.remove")
    def test_signal_based_add_and_remove_error_handlers(self, mock_remove, mock_store):
        """
        Errors are properly handled when add or remove signal is sent.
        """

        remove_handler = MagicMock()
        add_handler = MagicMock()

        mock_remove.side_effect = Exception()
        mock_store.side_effect = Exception()
        aapl = Stock(symbol="AAPL", name="Apple", market_cap=50)

        signal_registry.register(
            Stock, add_error_handler=add_handler, remove_error_handler=remove_handler
        )

        aapl.save()
        # There are 2 autocompleter providers for the Stock model. Therefore expect two error handler calls
        self.assertEqual(add_handler.call_count, 2)

        aapl.delete()
        self.assertEqual(remove_handler.call_count, 2)

        signal_registry.unregister(Stock)

    @patch("autocompleter.base.AutocompleterProviderBase.store")
    @patch("autocompleter.base.AutocompleterProviderBase.remove")
    def test_add_and_remove_error_handlers(self, mock_remove, mock_store):
        """
        Errors are properly handled when adding or removing manually
        """

        remove_handler = MagicMock()
        add_handler = MagicMock()

        mock_remove.side_effect = Exception()
        mock_store.side_effect = Exception()
        aapl = Stock(symbol="AAPL", name="Apple", market_cap=50)

        add_obj_to_autocompleter(
            Stock,
            aapl,
            False,
            add_error_handler=add_handler,
            remove_error_handler=remove_handler,
        )
        # There are 2 autocompleter providers for the Stock model. Therefore expect two error handler calls
        self.assertEqual(add_handler.call_count, 2)

        remove_obj_from_autocompleter(Stock, aapl, remove_error_handler=remove_handler)
        self.assertEqual(remove_handler.call_count, 2)

        signal_registry.unregister(Stock)

    @patch("autocompleter.base.AutocompleterProviderBase.store")
    @patch("autocompleter.base.AutocompleterProviderBase.remove")
    def test_unregister_removes_error_handlers(self, mock_remove, mock_store):
        """
        Unregistering removes registered error handling
        """
        remove_handler = MagicMock()
        add_handler = MagicMock()

        mock_remove.side_effect = Exception()
        mock_store.side_effect = Exception()
        aapl = Stock(symbol="AAPL", name="Apple", market_cap=50)

        signal_registry.register(
            Stock, add_error_handler=add_handler, remove_error_handler=remove_handler
        )

        # re register Stock without error handlers
        signal_registry.unregister(Stock)
        signal_registry.register(Stock)

        with self.assertRaises(Exception):
            aapl.save()

        with self.assertRaises(Exception):
            aapl.delete()
        signal_registry.unregister(Stock)

    def test_signal_based_update(self):
        """
        Turning on signals will automatically update objects in the autocompleter
        """
        signal_registry.register(Stock)

        aapl = Stock(symbol="AAPL", name="Apple", market_cap=50)
        aapl.save()

        autocomp = Autocompleter("stock")
        matches = autocomp.suggest("aapl")

        self.assertEqual(len(matches), 1)
        aapl.symbol = "XYZ"
        aapl.name = "XYZ & Co."
        aapl.save()

        matches = autocomp.suggest("aapl")
        self.assertEqual(len(matches), 0)
        matches = autocomp.suggest("xyz")
        self.assertEqual(len(matches), 1)

        aapl.delete()
        keys = self.redis.keys("djac.test.stock*")
        self.assertEqual(len(keys), 0)

        signal_registry.unregister(Stock)

    def test_register(self):
        """
        Register/Unregister works
        """
        registry.unregister("stock", StockAutocompleteProvider)
        providers = registry.get_all_by_autocompleter("stock")
        self.assertEqual(len(providers), 0)

        # Have to leave things the way models.py init-ed them for other tests!
        registry.register("stock", StockAutocompleteProvider)
        providers = registry.get_all_by_autocompleter("stock")
        self.assertEqual(len(providers), 1)


class UpdateTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json", "indicator_test_data_small.json"]

    def test_update_data(self):
        """
        Updating an obj's data updates the relevant redis objects
        """
        autocomp = Autocompleter("faceted_stock")
        autocomp.store_all()

        # Change AAPL's sector to 'Food'
        aapl = Stock.objects.get(symbol="AAPL")
        aapl.sector = "Food"
        aapl.save()

        autocomp.update_all(clear_cache=False)

        provider = FacetedStockAutocompleteProvider(aapl)
        expected_data = provider.get_data()

        data_map_key = base.AUTO_BASE_NAME % provider
        data_in_redis = autocomp._deserialize_data(
            self.redis.hget(data_map_key, provider.get_item_id())
        )
        self.assertEqual(expected_data, data_in_redis)
        self.assertEqual(data_in_redis["sector"], "Food")

    def test_update_facets(self):
        """
        Updating an obj's facets updates the relevant redis objects
        """
        autocomp = Autocompleter("faceted_stock")
        autocomp.store_all()

        # Change AAPL's sector to 'Food'
        aapl = Stock.objects.get(symbol="AAPL")
        aapl.sector = "Food"
        aapl.save()

        # Original facets are ['sector', 'industry'] and we're patching them to ['sector', 'search_name']
        with patch.object(
            FacetedStockAutocompleteProvider,
            "get_facets",
            return_value=["sector", "search_name"],
        ):
            autocomp.update_all(clear_cache=False)

        provider_name = FacetedStockAutocompleteProvider.get_provider_name()
        facet_set_key = base.FACET_SET_BASE_NAME % (provider_name, "{}", "{}")
        # AAPL is in djac.test.faceted_stock.sector.Food
        self.assertIsNotNone(
            self.redis.zscore(facet_set_key.format("sector", "Food"), aapl.id)
        )
        # AAPL is in djac.test.faceted_stock.search_symbol.AAPL
        self.assertIsNotNone(
            self.redis.zscore(facet_set_key.format("search_name", "AAPL"), aapl.id)
        )

        # AAPL is no longer in djac.test.faceted_stock.sector.Technology
        self.assertIsNone(
            self.redis.zscore(facet_set_key.format("sector", "Technology"), aapl.id)
        )
        # AAPL is no longer in djac.test.faceted_stock.industry.Consumer Electronics
        self.assertIsNone(
            self.redis.zscore(
                facet_set_key.format("industry", "Consumer Electronics"), aapl.id
            )
        )
        facet_map_key = base.FACET_MAP_BASE_NAME % (provider_name,)
        facet_data = autocomp._deserialize_data(self.redis.hget(facet_map_key, aapl.id))
        # Facet list was updated for AAPL in djac.test.faceted_stock
        self.assertEqual(
            facet_data,
            [
                {"key": "sector", "value": aapl.sector},
                {"key": "search_name", "value": aapl.symbol},
            ],
        )

    def test_terms_add_new_term(self):
        """
        Adding a new term to an object updates the relevant redis objects
        """
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 10)
        autocomp = Autocompleter("indicator_aliased")

        autocomp.store_all()

        provider_name = IndicatorAliasedAutocompleteProvider.get_provider_name()

        # Original name for indicator is "US Unemployment Rate"
        unemployment = Indicator.objects.get(internal_name="unemployment_rate")
        new_term = "xyzqwer"
        unemployment.name += f" {new_term}"
        unemployment.save()

        # Store a reference to previous norm terms, before updating
        original_norm_terms_in_db = Autocompleter._deserialize_data(
            self.redis.hget(base.TERM_MAP_BASE_NAME % provider_name, unemployment.id)
        )

        autocomp.update_all(clear_cache=False)

        # Verify that all new prefixes were added
        new_prefixes = [new_term[:x] for x in range(1, len(new_term) + 1)]
        for prefix in new_prefixes:
            # Obj id lives in djac.test.indal.p.[x, xy, xyz, xyzq, ...]
            prefix_key = base.PREFIX_BASE_NAME % (provider_name, prefix)
            self.assertIsNotNone(self.redis.zscore(prefix_key, unemployment.id))

            # The prefix is present in djac.test.indal.ps
            self.assertTrue(
                self.redis.sismember(base.PREFIX_SET_BASE_NAME % provider_name, prefix)
            )

        terms = IndicatorAliasedAutocompleteProvider(unemployment).get_terms()
        updated_norm_terms = IndicatorAliasedAutocompleteProvider._get_norm_terms(terms)
        # Verify that all new terms got inserted
        for term in updated_norm_terms:
            key = base.EXACT_BASE_NAME % (provider_name, term)
            self.assertIsNotNone(self.redis.zscore(key, unemployment.id))

        # Verify that all the old terms are no longer present
        for term in original_norm_terms_in_db:
            key = base.EXACT_BASE_NAME % (provider_name, term)
            self.assertIsNone(self.redis.zscore(key, unemployment.id))

        # Verify that the list of norm terms was updated in djac.test.indal
        norm_terms_in_db = Autocompleter._deserialize_data(
            self.redis.hget(base.TERM_MAP_BASE_NAME % provider_name, unemployment.id)
        )
        self.assertEqual(norm_terms_in_db, updated_norm_terms)

    def test_terms_remove_term(self):
        """
        Removing a term from an object updates the relevant redis objects
        """

        # Modify the indicator's name before storing
        unemployment = Indicator.objects.get(internal_name="unemployment_rate")
        original_name = unemployment.name
        new_term = "xyzqwer"
        unemployment.name += f" {new_term}"
        unemployment.save()

        # Store them all
        autocomp = Autocompleter("indicator_aliased")
        autocomp.store_all()

        # Update the indicator's name to its original name
        unemployment = Indicator.objects.get(internal_name="unemployment_rate")
        unemployment.name = original_name
        unemployment.save()

        # Update the autocompleter
        autocomp.update_all(clear_cache=False)

        provider_name = IndicatorAliasedAutocompleteProvider.get_provider_name()
        # Verify that all removed prefixes were deleted
        removed_prefixes = [new_term[:x] for x in range(1, len(new_term) + 1)]
        for prefix in removed_prefixes:
            # Obj id no longer lives in djac.test.indal.p.[x, xy, xyz, xyzq, ...]
            prefix_key = base.PREFIX_BASE_NAME % (provider_name, prefix)
            self.assertIsNone(self.redis.zscore(prefix_key, unemployment.id))

            # The prefix is not present in djac.test.indal.ps
            self.assertFalse(
                self.redis.sismember(base.PREFIX_SET_BASE_NAME % provider_name, prefix)
            )
        # Verify that the list of norm terms was updated in djac.test.indal
        norm_terms_in_db = Autocompleter._deserialize_data(
            self.redis.hget(base.TERM_MAP_BASE_NAME % provider_name, unemployment.id)
        )
        updated_norm_terms = IndicatorAliasedAutocompleteProvider._get_norm_terms(
            IndicatorAliasedAutocompleteProvider(unemployment).get_terms()
        )
        self.assertEqual(norm_terms_in_db, updated_norm_terms)

    def test_obj_deleted(self):
        # Store all objects
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 10)
        autocomp = Autocompleter("faceted_stock")
        autocomp.store_all()

        # Delete a stock
        aapl = Stock.objects.get(symbol="AAPL")
        obj_id = aapl.id
        aapl.delete()

        provider = FacetedStockAutocompleteProvider(aapl)
        provider_name = provider.get_provider_name()
        terms = provider._get_norm_terms(provider.get_terms())
        facets = provider.get_facets_dict()

        autocomp.update_all(clear_cache=False)

        exact_map_key = base.EXACT_SET_BASE_NAME % provider_name
        # Verify that exact terms are no longer present
        self.assertFalse(any(self.redis.smismember(exact_map_key, *terms)))
        for term in terms:
            key = base.EXACT_BASE_NAME % (provider_name, term)
            self.assertIsNone(self.redis.zscore(key, obj_id))

        # Prefixes unique to APPL are [ap, app, appl, apple]
        prefixes = {"apple"[:x] for x in range(2, len("apple") + 1)}
        prefixes_map_key = base.PREFIX_SET_BASE_NAME % provider_name
        # Verify that no prefixes are present
        self.assertFalse(any(self.redis.smismember(prefixes_map_key, *prefixes)))
        for prefix in prefixes:
            key = base.PREFIX_BASE_NAME % (provider_name, prefix)
            self.assertIsNone(self.redis.zscore(key, obj_id))

        # Verify that no facets are present
        facets_map_key = base.FACET_MAP_BASE_NAME % provider
        self.assertFalse(self.redis.hexists(facets_map_key, obj_id))
        for facet in facets:
            key = base.FACET_SET_BASE_NAME % (
                provider_name,
                facet["key"],
                facet["value"],
            )
            self.assertIsNone(self.redis.zscore(key, obj_id))

        # Terms are deleted as well from the term map
        term_map_key = base.TERM_MAP_BASE_NAME % provider_name
        self.assertFalse(self.redis.hexists(term_map_key, obj_id))

        # Data is no longer available
        data_map_key = base.AUTO_BASE_NAME % provider_name
        self.assertFalse(self.redis.hexists(data_map_key, obj_id))

        # Score is no longer in the hash map
        score_map_key = base.SCORE_MAP_BASE_NAME % provider_name
        self.assertFalse(self.redis.hexists(score_map_key, obj_id))

    def test_update_score(self):
        """ """
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 10)
        autocomp = Autocompleter("faceted_stock")
        autocomp.store_all()

        # Change AAPL's score, which is mapped to its market_cap
        new_score = 42
        aapl = Stock.objects.get(symbol="AAPL")
        aapl.market_cap = new_score
        aapl.save()

        expected_score = 1 / new_score

        provider = FacetedStockAutocompleteProvider(aapl)
        provider_name = provider.get_provider_name()
        obj_id = provider.get_item_id()

        autocomp.update_all(clear_cache=False)

        prefixes = []
        # New score updated in djac.test.faceted_stock.p.*
        for prefix in prefixes:
            prefix_key = base.PREFIX_BASE_NAME % (provider_name, prefix)
            self.assertEqual(self.redis.zscore(prefix_key, obj_id), expected_score)

        # New score updated in djac.test.faceted_stock.e.term
        terms = FacetedStockAutocompleteProvider._get_norm_terms(provider.get_terms())
        for term in terms:
            key = base.EXACT_BASE_NAME % (provider_name, term)
            self.assertEqual(self.redis.zscore(key, obj_id), expected_score)

        # New score updated in djac.test.faceted_stock.f.key.value
        facets = provider.get_facets_dict()
        for facet in facets:
            key = base.FACET_SET_BASE_NAME % (
                provider_name,
                facet["key"],
                facet["value"],
            )
            self.assertEqual(self.redis.zscore(key, obj_id), expected_score)

        # New score updated in djac.test.faceted_stock.sc
        self.assertEqual(
            Autocompleter._deserialize_data(
                self.redis.hget(base.SCORE_MAP_BASE_NAME % provider_name, obj_id)
            ),
            expected_score,
        )
