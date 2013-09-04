#!/usr/bin/python
# -*- coding: utf-8 -*-

from test_app.tests.base import AutocompleterTestCase
from test_app.models import Stock, Indicator, StockAutocompleteProvider, MetricAutocompleteProvider, IndicatorAutocompleteProvider
from test_app import calc_info
from autocompleter import Autocompleter, registry, signal_registry
from autocompleter import settings as auto_settings


class StoringAndRemovingTestCase(AutocompleterTestCase):
    fixtures = ['stock_test_data_small.json', 'indicator_test_data_small.json']

    def test_store_and_remove(self):
        """
        Storing and removing an item works
        """
        aapl = Stock.objects.get(symbol='AAPL')
        provider = StockAutocompleteProvider(aapl)

        provider.store()
        keys = self.redis.hkeys('djac.stock')
        self.assertEqual(len(keys), 1)

        provider.remove()
        keys = self.redis.keys('djac.stock*')
        self.assertEqual(len(keys), 0)

    def test_dict_store_and_remove(self):
        """
        Storing and removing an dictionary item works
        """
        item = calc_info.calc_dicts[0]
        provider = MetricAutocompleteProvider(item)
        provider.store()

        keys = self.redis.hkeys('djac.metric')
        self.assertEqual(len(keys), 1)

        provider.remove()
        keys = self.redis.keys('djac.metric*')
        self.assertEqual(len(keys), 0)

    def test_store_and_remove_all_basic(self):
        """
        Storing and removing items all at once works for a dictionary obj autocompleter.
        """
        autocomp = Autocompleter("stock")

        autocomp.store_all()
        keys = self.redis.hkeys('djac.stock')
        self.assertEqual(len(keys), 101)

        autocomp.remove_all()
        keys = self.redis.keys('djac.stock*')
        self.assertEqual(len(keys), 0)

    def test_oprhan_removal(self):
        """
        test orphan removal
        """
        signal_registry.register(Indicator)

        autocomp = Autocompleter("indicator")
        autocomp.store_all()

        unemployment = Indicator.objects.get(internal_name='unemployment_rate')

        unemployment.name = 'free parking'
        unemployment.save()


        self.assertTrue(autocomp.suggest('free parking')[0]['id'] == 1)
        self.assertTrue(autocomp.suggest('US Unemployment Rate')[0]['id'] == 1)

        IndicatorAutocompleteProvider.delete_old_terms('US Unemployment Rate')
        self.assertTrue(autocomp.suggest('free parking')[0]['id'] == 1)
        self.assertTrue(len(autocomp.suggest('US Unemployment Rate')) == 0)

        signal_registry.unregister(Indicator)
        autocomp.remove_all()

    def test_dict_store_and_remove_all_basic(self):
        """
        Storing and removing items all at once works for a single-model autocompleter.
        """
        autocomp = Autocompleter("metric")

        autocomp.store_all()
        keys = self.redis.hkeys('djac.metric')
        self.assertEqual(len(keys), 6)

        autocomp.remove_all()
        keys = self.redis.keys('djac.metric')
        self.assertEqual(len(keys), 0)

    def test_store_and_remove_all_basic_with_caching(self):
        """
        Storing and removing items all at once works with caching turned on
        """
        # Let's turn on caching because that will store things in Redis and we want to make
        # sure we clean them up.
        setattr(auto_settings, 'CACHE_TIMEOUT', 3600)

        autocomp = Autocompleter("stock")
        autocomp.store_all()

        keys = self.redis.hkeys('djac.stock')
        self.assertEqual(len(keys), 101)

        autocomp = Autocompleter("stock")
        for i in range(0, 3):
            autocomp.suggest('a')
            autocomp.suggest('z')
            autocomp.exact_suggest('aapl')
            autocomp.exact_suggest('xyz')

        autocomp.remove_all()
        keys = self.redis.keys('djac.stock*')
        self.assertEqual(len(keys), 0)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, 'CACHE_TIMEOUT', 0)

    def test_dict_store_and_remove_all_basic_with_caching(self):
        """
        Storing and removing items all at once works with caching turned on on dict ac
        """
        # Let's turn on caching because that will store things in Redis and we want to make
        # sure we clean them up.
        setattr(auto_settings, 'CACHE_TIMEOUT', 3600)

        autocomp = Autocompleter("metric")
        autocomp.store_all()

        keys = self.redis.hkeys("djac.metric")
        self.assertEqual(len(keys), 6)

        autocomp = Autocompleter("metric")
        for i in range(0, 3):
            autocomp.suggest('m')
            autocomp.suggest('e')
            autocomp.exact_suggest('PE Ratio TTM')
            autocomp.exact_suggest('Market Cap')

        autocomp.remove_all()

        keys = self.redis.keys('djac.metric*')
        self.assertEqual(len(keys), 0)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, 'CACHE_TIMEOUT', 0)

    def test_store_and_remove_all_multi(self):
        """
        Storing and removing items all at once works for a multi-model autocompleter.
        """
        autocomp = Autocompleter("mixed")

        autocomp.store_all()
        keys = self.redis.hkeys('djac.stock')
        self.assertEqual(len(keys), 101)
        keys = self.redis.hkeys('djac.ind')
        self.assertEqual(len(keys), 100)
        keys = self.redis.hkeys("djac.metric")
        self.assertEqual(len(keys), 6)

        autocomp.remove_all()
        keys = self.redis.keys('djac.stock*')
        self.assertEqual(len(keys), 0)
        keys = self.redis.keys('djac.ind*')
        self.assertEqual(len(keys), 0)
        keys = self.redis.keys('djac.mixed*')
        self.assertEqual(len(keys), 0)
        keys = self.redis.keys('djac.metric*')
        self.assertEqual(len(keys), 0)


class SelectiveStoringTestCase(AutocompleterTestCase):
    fixtures = ['indicator_test_data_small.json']

    def test_selective_add_and_remove(self):
        """
        We can exclude certain objects from the autocompleter selectively.
        """
        autocomp = Autocompleter("indicator")
        autocomp.store_all()
        matches = autocomp.suggest('us unemployment rate')
        self.assertEqual(len(matches), 1)
        autocomp.remove_all()

        autocomp = Autocompleter("indicator_selective")
        autocomp.store_all()
        matches = autocomp.suggest('us unemployment rate')
        self.assertEqual(len(matches), 0)
        autocomp.remove_all()


class SignalBasedStoringTestCase(AutocompleterTestCase):
    def test_signal_based_add_and_remove(self):
        """
        Turning on signals will automatically add and remove and object from the autocompleter
        """
        aapl = Stock(symbol='AAPL', name='Apple', market_cap=50)
        aapl.save()
        keys = self.redis.keys('djac.stock*')
        self.assertEqual(len(keys), 0)

        signal_registry.register(Stock)

        aapl.save()
        keys = self.redis.keys('djac.stock*')
        self.assertNotEqual(len(keys), 0)

        aapl.delete()
        keys = self.redis.keys('djac.stock*')
        self.assertEqual(len(keys), 0)

        signal_registry.unregister(Stock)

    def test_signal_based_update(self):
        """
        Turning on signals will automatically update objects in the autocompleter
        """
        signal_registry.register(Stock)

        aapl = Stock(symbol='AAPL', name='Apple', market_cap=50)
        aapl.save()

        autocomp = Autocompleter("stock")
        matches = autocomp.suggest('aapl')

        self.assertEqual(len(matches), 1)
        aapl.symbol = 'XYZ'
        aapl.name = 'XYZ & Co.'
        aapl.save()

        matches = autocomp.suggest('aapl')
        self.assertEqual(len(matches), 0)
        matches = autocomp.suggest('xyz')
        self.assertEqual(len(matches), 1)

        aapl.delete()
        keys = self.redis.keys('djac.stock*')
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
