import redis

from django_nose import FastFixtureTestCase
from django.conf import settings

from autocompleter import registry


class AutocompleterTestCase(FastFixtureTestCase):
    def setUp(self):
        self.redis = redis.Redis(host=settings.AUTOCOMPLETER_REDIS_CONNECTION['host'],
            port=settings.AUTOCOMPLETER_REDIS_CONNECTION['port'],
            db=settings.AUTOCOMPLETER_REDIS_CONNECTION['db'])

        # purge any possible old test data in case of previous failures where tearDown didn't fire.
        # This is hardcoded so you don't accidentally wipe your redis db somehow.
        old_data = self.redis.keys("djac.test.*")
        pipe = self.redis.pipeline()
        for i in old_data:
            pipe.delete(i)
        pipe.execute()

    def _get_provider_class(self, name, model):
        provider_classes = registry.get_all_by_autocompleter(name)
        for provider_class in provider_classes:
            if provider_class.model == model:
                return provider_class
        return None
