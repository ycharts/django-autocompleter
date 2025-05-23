#!/usr/bin/python
# -*- coding: utf-8 -*-

from test_app.autocompleters import (
    StockAutocompleteProvider,
    IndicatorAutocompleteProvider,
    CalcAutocompleteProvider,
)
from test_app.models import Stock
from test_app.tests.base import AutocompleterTestCase

from autocompleter import Autocompleter, registry
from autocompleter import settings as auto_settings


class StockMatchTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json"]

    def setUp(self):
        super(StockMatchTestCase, self).setUp()
        self.autocomp = Autocompleter("stock")
        self.autocomp.store_all()

    def tearDown(self):
        self.autocomp.remove_all()

    def test_simple_match(self):
        """
        Basic matching works
        """
        matches_symbol = self.autocomp.suggest("a")
        self.assertEqual(len(matches_symbol), 10)

    def test_no_match(self):
        """
        Phrases that match nothing work
        """
        matches_symbol = self.autocomp.suggest("gobblygook")
        self.assertEqual(len(matches_symbol), 0)

    def test_dual_term_matches(self):
        """
        Items in autocompleter can match against multiple unique terms
        """
        matches_symbol = self.autocomp.suggest("AAPL")
        self.assertEqual(len(matches_symbol), 1)

        matches_name = self.autocomp.suggest("Apple")
        self.assertEqual(len(matches_name), 1)

    def test_accented_matches(self):
        """
        Accented phrases match against both their orignal accented form, and their non-accented basic form.
        """
        matches = self.autocomp.suggest("estee lauder")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["search_name"], "EL")

        matches = self.autocomp.suggest("estée lauder")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["search_name"], "EL")

    def test_max_results_setting(self):
        """
        MAX_RESULTS is respected.
        """
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 10)
        setattr(auto_settings, "MAX_RESULTS", 2)
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 2)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "MAX_RESULTS", 10)

    def test_ac_specific_max_results_setting(self):
        """
        Autocompleter specific MAX_RESULTS is respected
        """
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 10)

        registry.set_autocompleter_setting("stock", "MAX_RESULTS", 5)
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 5)

        # Must set the setting back to where it was as it will persist
        registry.del_autocompleter_setting("stock", "MAX_RESULTS")

    def test_caching(self):
        """
        Caching works
        """
        matches = self.autocomp.suggest("a")

        setattr(auto_settings, "CACHE_TIMEOUT", 3600)

        for i in range(0, 3):
            matches2 = self.autocomp.suggest("a")

        self.assertEqual(len(matches), len(matches2))

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "CACHE_TIMEOUT", 0)

    def test_dropped_character_matching(self):
        """
        Searching for things that would be normalized to ' ' do not
        result in redis errors.
        """
        matches = self.autocomp.suggest("+")
        self.assertEqual(len(matches), 0)
        matches = self.autocomp.suggest(
            "NBBC vs Regional - Mid-Atlantic Banks vs Financial"
        )
        self.assertEqual(len(matches), 0)


class IndicatorMatchTestCase(AutocompleterTestCase):
    fixtures = ["indicator_test_data_small.json"]

    def setUp(self):
        super().setUp()
        self.autocomp = Autocompleter("indicator")
        self.autocomp.store_all()

    def tearDown(self):
        self.autocomp.remove_all()

    def test_same_score_word_based_id_ordering(self):
        """
        Two results with the same score are returned in lexographic order of object ID
        """

        matches = self.autocomp.suggest("us")
        self.assertEqual(
            matches[1]["display_name"], "US Dollar to Australian Dollar Exchange Rate"
        )
        self.assertEqual(
            matches[9]["display_name"], "US Dollar to Chinese Yuan Exchange Rate"
        )
        return matches

    def test_join_char_replacement(self):
        """
        Dashes are handled correctly
        """
        # Testing that both '3-month' and '3 month' match
        matches = self.autocomp.suggest("3-month")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("3 month")
        self.assertNotEqual(len(matches), 0)

        matches = self.autocomp.suggest("mortgage-backed")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("mortgagebacked")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("mortgage backed")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("backed mortgage")
        self.assertNotEqual(len(matches), 0)

        matches = self.autocomp.suggest("U S A")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("U SA")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("USA")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("U-S-A")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("U/S/A")
        self.assertNotEqual(len(matches), 0)
        matches = self.autocomp.suggest("U-S/A")
        self.assertNotEqual(len(matches), 0)

    def test_min_letters_setting(self):
        """
        MIN_LETTERS is respected.
        """
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 10)
        setattr(auto_settings, "MIN_LETTERS", 2)
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 0)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "MIN_LETTERS", 1)

    def test_ac_provider_specific_min_letters_setting(self):
        """
        Autocompleter/Provider specific MIN_LETTERS is respected.
        """
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 10)
        setattr(auto_settings, "MIN_LETTERS", 2)
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 0)

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "MIN_LETTERS", 1)


class DictProviderMatchingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json"]

    def setUp(self):
        super(DictProviderMatchingTestCase, self).setUp()
        self.autocomp = Autocompleter("metric")
        self.autocomp.store_all()

    def tearDown(self):
        self.autocomp.remove_all()

    def test_basic_match(self):
        matches = self.autocomp.suggest("m")
        self.assertEqual(len(matches), 1)


class MultiMatchingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json", "indicator_test_data_small.json"]

    def setUp(self):
        super(MultiMatchingTestCase, self).setUp()
        self.autocomp = Autocompleter("mixed")
        self.autocomp.store_all()

    def tearDown(self):
        self.autocomp.remove_all()

    def test_basic_match(self):
        """
        A single autocompleter can return results from multiple models.
        """
        matches = self.autocomp.suggest("Aapl")
        self.assertEqual(len(matches["stock"]), 1)

        matches = self.autocomp.suggest("US Initial Claims")
        self.assertEqual(len(matches["ind"]), 1)

        matches = self.autocomp.suggest("m")
        self.assertEqual(len(matches), 3)

        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches["stock"]), 6)
        self.assertEqual(len(matches["ind"]), 4)

    def test_min_letters_setting(self):
        """
        MIN_LETTERS is respected in multi-type search case.
        """
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches["stock"]), 6)
        self.assertEqual(len(matches["ind"]), 4)

        setattr(auto_settings, "MIN_LETTERS", 2)
        matches = self.autocomp.suggest("a")
        self.assertEqual(matches, {})

        setattr(auto_settings, "MIN_LETTERS", 1)

    def test_ac_provider_specific_min_letters_setting(self):
        """
        Autocompleter/Provider specific MIN_LETTERS is respected in multi-type search case.
        """
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches["stock"]), 6)
        self.assertEqual(len(matches["ind"]), 4)

        registry.set_ac_provider_setting(
            "mixed", IndicatorAutocompleteProvider, "MIN_LETTERS", 2
        )
        registry.set_ac_provider_setting(
            "mixed", CalcAutocompleteProvider, "MIN_LETTERS", 2
        )
        matches = self.autocomp.suggest("a")
        self.assertEqual(len(matches), 10)
        self.assertEqual("ind" not in matches, True)

        registry.del_ac_provider_setting(
            "mixed", IndicatorAutocompleteProvider, "MIN_LETTERS"
        )
        registry.del_ac_provider_setting(
            "mixed", CalcAutocompleteProvider, "MIN_LETTERS"
        )


class MaxResultsMatchingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json", "indicator_test_data_small.json"]

    def setUp(self):
        super(MaxResultsMatchingTestCase, self).setUp()
        self.autocomp = Autocompleter("ind_stock")
        self.autocomp.store_all()

    def test_max_results_respected(self):
        """
        MAX_RESULTS is respected for multi-type search case
        """
        # set MAX_RESULTS to an arbitrarily large number
        registry.set_autocompleter_setting("ind_stock", "MAX_RESULTS", 100)

        matches = self.autocomp.suggest("a")
        total_matches_with_large_max_results = len(matches["stock"]) + len(
            matches["ind"]
        )
        self.assertGreaterEqual(100, total_matches_with_large_max_results)
        self.assertEqual(41, total_matches_with_large_max_results)

        registry.set_autocompleter_setting("ind_stock", "MAX_RESULTS", 4)
        matches = self.autocomp.suggest("a")
        total_matches_with_small_max_results = len(matches["stock"]) + len(
            matches["ind"]
        )
        self.assertEqual(4, total_matches_with_small_max_results)

        self.assertGreater(
            total_matches_with_large_max_results, total_matches_with_small_max_results
        )

        registry.del_autocompleter_setting("ind_stock", "MAX_RESULTS")

    def test_max_results_spreads_results_evenly(self):
        """
        MAX_RESULTS spreads the results among providers equally
        """
        registry.set_autocompleter_setting("ind_stock", "MAX_RESULTS", 4)
        matches = self.autocomp.suggest("a")
        self.assertEqual(4, len(matches["stock"]) + len(matches["ind"]))
        self.assertEqual(len(matches["stock"]), len(matches["ind"]))

        registry.del_autocompleter_setting("ind_stock", "MAX_RESULTS")

    def test_max_results_handles_surplus(self):
        """
        Suggest respects MAX_RESULTS while still dealing with surplus
        """
        # we know that there are 16 ind matches and 25 stock matches for 'a'
        registry.set_autocompleter_setting("ind_stock", "MAX_RESULTS", 36)
        matches = self.autocomp.suggest("a")
        self.assertEqual(16, len(matches["ind"]))
        self.assertEqual(20, len(matches["stock"]))

        registry.del_autocompleter_setting("ind_stock", "MAX_RESULTS")

    def test_max_results_handles_deficit_less_than_surplus(self):
        """
        MAX_RESULTS stops trying to hand out surplus matches when provider's deficits are met
        """
        # Previous code would have failed on this test case because of an infinite while loop that
        # failed to break when all provider's deficits were met. The setup requires
        # there to be at least one provider which will have a deficit less than the total surplus.
        # In this test case, the total surplus will be 3 (since stock has no matches) and the deficit
        # for indicators will be 2 (since there are 5 total matches for indicators and 3 slots are
        # reserved initially)
        registry.set_autocompleter_setting("ind_stock", "MAX_RESULTS", 6)
        matches = self.autocomp.suggest("S&P")
        self.assertEqual(5, len(matches["ind"]))
        self.assertEqual(0, len(matches["stock"]))

        registry.del_autocompleter_setting("ind_stock", "MAX_RESULTS")

    def test_max_results_has_hard_limit(self):
        """
        Suggest respects MAX_RESULTS over giving every provider at least 1 result
        """
        registry.set_autocompleter_setting("ind_stock", "MAX_RESULTS", 1)
        matches = self.autocomp.suggest("a")
        # Either stock or indicator matches is empty
        self.assertEqual(1, len(matches["stock"]) + len(matches["ind"]))

        registry.del_autocompleter_setting("ind_stock", "MAX_RESULTS")


class FacetMatchingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json"]

    def setUp(self):
        super(FacetMatchingTestCase, self).setUp()
        self.autocomp = Autocompleter("faceted_stock")
        self.autocomp.store_all()

    def test_facet_or_match(self):
        """
        Matching using facets works with the 'or' type
        """
        facets = [
            {
                "type": "or",
                "facets": [
                    {"key": "sector", "value": "Technology"},
                    {"key": "sector", "value": "Consumer Defensive"},
                    {"key": "industry", "value": "Thisdoesntexist"},
                ],
            }
        ]
        matches = self.autocomp.suggest("a", facets=facets)
        self.assertEqual(len(matches), 4)

    def test_facet_and_match(self):
        """
        Matching using facets works with the 'and' type
        """
        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "sector", "value": "Technology"},
                    {"key": "industry", "value": "SectorDoesntExist"},
                ],
            }
        ]
        matches = self.autocomp.suggest("ch", facets=facets)
        # since a stock can't have two different values for a sector, we expect calculating an 'and' on
        # two different sectors to have zero matches
        self.assertEqual(len(matches), 0)

        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "sector", "value": "Communication Services"},
                    {"key": "industry", "value": "Telecom Services"},
                ],
            }
        ]

        matches = self.autocomp.suggest("ch", facets=facets)
        self.assertEqual(len(matches), 1)

        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "sector", "value": "Energy"},
                    {"key": "industry", "value": "Oil & Gas Integrated"},
                ],
            }
        ]
        matches = self.autocomp.suggest("ch", facets=facets)
        self.assertEqual(len(matches), 2)

    def test_provider_keys_is_not_subset_of_facet_keys(self):
        """
        A provider's facet keys has to match at least one of the requested facet keys to kick in
        """

        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "thisisfake", "value": "Technology"},
                ],
            }
        ]
        facet_matches = self.autocomp.suggest("a", facets=facets)
        regular_matches = self.autocomp.suggest("a")
        # since the 'thisisfake' key does not exist in our provider, the results for a facet
        # suggest should be the same as a regular suggest
        self.assertEqual(facet_matches, regular_matches)

    def test_provider_keys_is_subset_of_facet_keys_no_match(self):
        """
        A provider which declares one of the requested facet keys but has no matches should not return any results
        """

        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "sector", "value": "ZZ9 Plural Z Alpha"},
                ],
            }
        ]
        facet_matches = self.autocomp.suggest("a", facets=facets)
        regular_matches = self.autocomp.suggest("a")
        # since the 'sector' key does belong to our faceted stock provider facets, we expected facets
        # logic to kick in, but with a bogus value there should be no results.
        self.assertEqual(len(facet_matches), 0)
        self.assertGreaterEqual(len(regular_matches), 1)

    def test_facet_doesnt_skew_suggest(self):
        """
        Test that using facets takes the suggest term into consideration
        """

        matches = self.autocomp.suggest("nosearchresultsforthisterm")
        self.assertEqual(len(matches), 0)
        facets = [
            {
                "type": "or",
                "facets": [
                    {"key": "sector", "value": "Technology"},
                ],
            }
        ]

        # we expect that adding facets to a suggest call with no results will not
        # add any results
        facet_matches = self.autocomp.suggest(
            "nosearchresultsforthisterm", facets=facets
        )
        self.assertEqual(len(facet_matches), 0)

    def test_facet_match_with_move_exact_matches(self):
        """
        Exact matching still works with facet suggest
        """
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 10)
        temp_autocomp = Autocompleter("faceted_stock")
        temp_autocomp.store_all()

        facets = [
            {
                "type": "or",
                "facets": [
                    {"key": "sector", "value": "Technology"},
                    {"key": "industry", "value": "Software"},
                ],
            }
        ]

        matches = temp_autocomp.suggest("Ma", facets=facets)
        setattr(auto_settings, "MOVE_EXACT_MATCHES_TO_TOP", True)
        matches2 = temp_autocomp.suggest("Ma", facets=facets)
        self.assertNotEqual(matches[0]["search_name"], matches2[0]["search_name"])

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "MOVE_EXACT_MATCHES_TO_TOP", False)
        temp_autocomp.remove_all()

    def test_facet_mismatch_with_move_exact_matches(self):
        """
        Exact matching shouldn't move an object that doesn't have a matching facet value
        """
        # This test case depends on very specific data, which is why this test
        # issues multiple asserts to check our assumptions

        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 10)
        temp_autocomp = Autocompleter("faceted_stock")
        temp_autocomp.store_all()

        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "sector", "value": "Healthcare"},
                    {"key": "industry", "value": "Healthcare Plans"},
                ],
            }
        ]

        # When gathering suggestions for 'Un', based on the stock_data_small.json fixture,
        # the only match should be UnitedHealth Group Inc. when using the Healthcare sector facet
        matches = temp_autocomp.suggest("Un", facets=facets)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["search_name"], "UNH")

        # When MOVE_EXACT_MATCHES_TO_TOP is set to True and not using facets,
        # we are expecting Unilever to be moved to the top.
        setattr(auto_settings, "MOVE_EXACT_MATCHES_TO_TOP", True)
        matches = temp_autocomp.suggest("Un")
        self.assertEqual(matches[0]["search_name"], "UN")

        # When MOVE_EXACT_MATCHES_TO_TOP is set to True and we are using the
        # Healthcare sector facet, we are expecting to see UnitedHealth group
        # since Unilever belongs to the Consumer Defensive sector
        matches = temp_autocomp.suggest("Un", facets=facets)
        self.assertEqual(matches[0]["search_name"], "UNH")

        # Must set the setting back to where it was as it will persist
        setattr(auto_settings, "MOVE_EXACT_MATCHES_TO_TOP", False)
        temp_autocomp.remove_all()

    def test_exact_match_low_score_still_at_top(self):
        """
        Exact matching when using facets should push low scoring object to top if exact match
        """
        # The setup for this test case is that we have three stocks that begin with the letter Z
        # but limit our MAX_RESULTS setting to just 2.
        # With MOVE_EXACT_MATCHES_TO_TOP initially set to False, we do not expect to see the
        # stock with symbol 'Z' in the suggest results since it has the lowest market cap of the 3 which is
        # what we base the score off of.
        # Once we set MOVE_EXACT_MATCHES_TO_TOP to True we expect Z to be right at the top even though
        # it didn't even show up in the previous suggest call since it is an exact match.
        setattr(auto_settings, "MAX_RESULTS", 2)
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 10)
        temp_autocomp = Autocompleter("faceted_stock")
        temp_autocomp.store_all()
        facets = [
            {
                "type": "and",
                "facets": [
                    {"key": "sector", "value": "Technology"},
                    {"key": "industry", "value": "Software"},
                ],
            }
        ]
        matches = temp_autocomp.suggest("Z", facets=facets)
        search_names = [match["search_name"] for match in matches]
        self.assertTrue("Z" not in search_names)

        setattr(auto_settings, "MOVE_EXACT_MATCHES_TO_TOP", True)
        matches = temp_autocomp.suggest("Z", facets=facets)
        self.assertTrue(matches[0]["search_name"] == "Z")

        setattr(auto_settings, "MOVE_EXACT_MATCHES_TO_TOP", False)
        setattr(auto_settings, "MAX_RESULTS", 10)
        setattr(auto_settings, "MAX_EXACT_MATCH_WORDS", 0)
        temp_autocomp.remove_all()

    def test_facet_works_with_cache(self):
        """
        Caching works with facet suggest
        """
        no_cache_no_facet_matches = self.autocomp.suggest("a")

        facets = [{"type": "or", "facets": [{"key": "sector", "value": "Technology"}]}]

        no_cache_facet_matches = self.autocomp.suggest("a", facets=facets)

        setattr(auto_settings, "CACHE_TIMEOUT", 3600)

        cache_facet_matches = self.autocomp.suggest("a", facets=facets)
        cache_no_facet_matches = self.autocomp.suggest("a")

        self.assertEqual(no_cache_no_facet_matches, cache_no_facet_matches)
        self.assertEqual(cache_facet_matches, no_cache_facet_matches)

        setattr(auto_settings, "CACHE_TIMEOUT", 0)

    def test_multiple_facet_dicts_match(self):
        """
        Matching with multiple passed in facet dicts works
        """
        facets = [
            {
                "type": "and",
                "facets": [{"key": "sector", "value": "Communication Services"}],
            },
            {
                "type": "and",
                "facets": [{"key": "industry", "value": "Telecom Services"}],
            },
        ]
        matches = self.autocomp.suggest("ch", facets=facets)
        self.assertEqual(len(matches), 1)

        facets = [
            {"type": "and", "facets": [{"key": "sector", "value": "Energy"}]},
            {
                "type": "and",
                "facets": [{"key": "industry", "value": "Oil & Gas Integrated"}],
            },
        ]
        matches = self.autocomp.suggest("ch", facets=facets)
        self.assertEqual(len(matches), 2)


class MixedFacetProvidersMatchingTestCase(AutocompleterTestCase):
    fixtures = ["stock_test_data_small.json", "indicator_test_data_small.json"]

    def setUp(self):
        super(MixedFacetProvidersMatchingTestCase, self).setUp()
        self.autocomp = Autocompleter("facet_stock_no_facet_ind")
        self.autocomp.store_all()

    def test_autocompleter_with_facet_and_non_facet_providers(self):
        """
        Autocompleter with facet and non-facet providers works correctly
        """
        registry.set_autocompleter_setting(
            "facet_stock_no_facet_ind", "MAX_RESULTS", 100
        )
        facets = [
            {
                "type": "and",
                "facets": [{"key": "sector", "value": "Financial Services"}],
            }
        ]
        matches = self.autocomp.suggest("a")
        facet_matches = self.autocomp.suggest("a", facets=facets)

        # because we are using the faceted stock provider in the 'facet_stock_no_facet_ind' AC,
        # we expect using facets will decrease the amount of results when searching.
        self.assertEqual(len(matches["faceted_stock"]), 25)
        self.assertEqual(len(facet_matches["faceted_stock"]), 2)

        # since the indicator provider does not support facets,
        # we expect the search results from both a facet and non-facet search to be the same.
        self.assertEqual(len(matches["ind"]), 16)
        self.assertEqual(len(matches["ind"]), len(facet_matches["ind"]))

        registry.del_autocompleter_setting("facet_stock_no_facet_ind", "MAX_RESULTS")
