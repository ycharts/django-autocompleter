from django.conf import settings

# GLOBAL SETTINGS #

# Number of seconds to cache results. 0 means no caching
CACHE_TIMEOUT = getattr(settings, "AUTOCOMPLETER_CACHE_TIMEOUT", 0)

# Regex that filters out characters we ignore for the purposes of autocompleting
CHARACTER_FILTER = getattr(settings, "AUTOCOMPLETER_CHARACTER_FILTER", r"[^a-z0-9_ ]")

# Name of variable autocompleter will look for to populate facet data on a suggest call
FACET_PARAMETER_NAME = getattr(settings, "AUTOCOMPLETER_FACET_PARAMETER_NAME", "facets")

# When an autocompleter only has one type of result it can return, this setting determines
# whether the results set is "flattened" so that it no longer uses the data structure needed
# to return multi-type results
FLATTEN_SINGLE_TYPE_RESULTS = getattr(
    settings, "AUTOCOMPLETER_FLATTEN_SINGLE_TYPE_RESULTS", True
)

# Characters we want the autocompleter to interpret as both a space and a blank string.
# Meaning by default, 'U/S-A' will also be stored as 'U SA', 'US A', 'U S A', and 'USA'
JOIN_CHARS = getattr(settings, "AUTOCOMPLETER_JOIN_CHARS", ["-", "/"])

# Redis connection parameters
REDIS_CONNECTION = getattr(settings, "AUTOCOMPLETER_REDIS_CONNECTION", {})

# Name of variable autocompleter will look for to grab what term to search on.
SUGGEST_PARAMETER_NAME = getattr(settings, "AUTOCOMPLETER_SUGGEST_PARAMETER_NAME", "q")

# Test data for debugging/running tests
TEST_DATA = getattr(settings, "AUTOCOMPLETER_TEST_DATA", False)

## Advanced optimization settings

# When handling a suggest call that involves a multi-word term that could lead to large result sizes,
# if the smallest set involved exceeds this threshold, skip the intersection operation
# and use that smallest set as a proxy for the intersection.
CARDINALITY_THRESHOLD_INTERSECTION = getattr(settings, "AUTOCOMPLETER_CARDINALITY_THRESHOLD_INTERSECTION", 50_000)

# When combining result sets across normalized term variations, if the combined number of
# members across all sets exceeds this threshold, skip the union and use the smallest
# set as a proxy for the union.
CARDINALITY_THRESHOLD_UNION = getattr(settings, "AUTOCOMPLETER_CARDINALITY_THRESHOLD_UNION", 50_000)

# AC SETTINGS #

# Maximum number of results returned per result type
MAX_RESULTS = getattr(settings, "AUTOCOMPLETER_MAX_RESULTS", 10)

# Whether to detect exact matches and move them to top of the results set (ignoring score)
# This will obviously not work if MAX_EXACT_MATCH_WORDS == 0 for your install or your provider.
MOVE_EXACT_MATCHES_TO_TOP = getattr(
    settings, "AUTOCOMPLETER_MOVE_EXACT_MATCHES_TO_TOP", False
)


# PROVIDER SETTINGS #

# Maximum number of words in term we should be able to match as exact match. Default is 0,
# which means there is no exact matching at all.
MAX_EXACT_MATCH_WORDS = getattr(settings, "AUTOCOMPLETER_MAX_EXACT_MATCH_WORDS", 0)


# AC/PROVIDER SETTINGS #

# Minimum number of letters required to start returning results
MIN_LETTERS = getattr(settings, "AUTOCOMPLETER_MIN_LETTERS", 1)
