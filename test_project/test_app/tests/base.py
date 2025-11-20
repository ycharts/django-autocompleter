import redis
from django.conf import settings
from django.core import management
from django.test import TestCase


class AutocompleterTestCase(TestCase):
    def setUp(self):
        # TODO: automatically test both cluster and non-cluster modes
        CLUSTER_MODE = settings.AUTOCOMPLETER_REDIS_CLUSTER_MODE
        if CLUSTER_MODE:
            from redis.cluster import RedisCluster as Redis
        else:
            from redis import Redis

        self.redis = Redis(
            host=settings.AUTOCOMPLETER_REDIS_CONNECTION["host"],
            port=settings.AUTOCOMPLETER_REDIS_CONNECTION["port"],
        )

    def tearDown(self):
        # Purge any possible old test data
        self.redis.flushdb()

    @classmethod
    def tearDownClass(cls):
        super(AutocompleterTestCase, cls).tearDownClass()
        management.call_command("flush", verbosity=0, interactive=False)

    @staticmethod
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]
