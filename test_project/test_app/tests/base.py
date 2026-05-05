import redis
from django.conf import settings
from django.core import management
from django.test import TestCase

from autocompleter import registry


class AutocompleterTestCase(TestCase):
    def setUp(self):
        self.redis = redis.Redis(
            host=settings.AUTOCOMPLETER_REDIS_CONNECTION["host"],
            port=settings.AUTOCOMPLETER_REDIS_CONNECTION["port"],
            db=settings.AUTOCOMPLETER_REDIS_CONNECTION["db"],
        )

    def tearDown(self):
        # Purge any possible old test data
        old_data = self.redis.keys("djac.test.*")
        pipe = self.redis.pipeline()
        for i in self.chunk_list(old_data, 100):
            pipe.delete(*i)
        pipe.execute()

    @classmethod
    def tearDownClass(cls):
        super(AutocompleterTestCase, cls).tearDownClass()
        management.call_command("flush", verbosity=0, interactive=False)

    @staticmethod
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    @staticmethod
    def store_all_for_ac(ac_name):
        for provider_class in registry.get_all_by_autocompleter(ac_name) or []:
            provider_class.store_all()

    @staticmethod
    def remove_all_for_ac(ac_name):
        for provider_class in registry.get_all_by_autocompleter(ac_name) or []:
            provider_class.remove_all()

    @staticmethod
    def update_all_for_ac(ac_name):
        for provider_class in registry.get_all_by_autocompleter(ac_name) or []:
            provider_class.update_all()
