from django.db.models.signals import post_save, pre_save, post_delete

from autocompleter import settings


class AutocompleterRegistry(object):
    def __init__(self):
        self._providers_by_ac = {}
        self._providers_by_model = {}
        self._ac_provider_settings = {}

    def register(self, ac_name, provider, local_settings=None):
        """
        Register an autocompleter wit ha  provider.
        Each autocompleter can have multiple providers.
        """
        if provider is None:
            return

        if ac_name not in self._providers_by_ac:
            self._providers_by_ac[ac_name] = []
        if provider.model not in self._providers_by_model:
            self._providers_by_model[provider.model] = []

        if provider not in self._providers_by_ac[ac_name]:
            self._providers_by_ac[ac_name].append(provider)
        if provider not in self._providers_by_model[provider.model]:
            self._providers_by_model[provider.model].append(provider)

        combined_name = "%s%s" % (ac_name, provider,)
        # Note: the reason we default local_settings to None, then set to a dict is when we had
        # local_settings default to {} it was a reference to the same dict so when a setting
        # for one AC/provider was set, it was set for all AC/provider pairs.
        if local_settings is None:
            local_settings = {}
        self._ac_provider_settings[combined_name] = local_settings

    def unregister(self, ac_name, provider):
        """
        Unregister a provider from the autocompleter.
        """
        if provider is None:
            return
        if ac_name in self._providers_by_ac and \
                provider in self._providers_by_ac[ac_name]:
            self._providers_by_ac[ac_name].remove(provider)
        if provider.model in self._providers_by_model and \
                provider in self._providers_by_model[provider.model]:
            self._providers_by_model[provider.model].remove(provider)

        combined_name = "%s%s" % (ac_name, provider,)
        del self._ac_provider_settings[combined_name]

    def get_all_by_autocompleter(self, ac_name):
        if ac_name not in self._providers_by_ac:
            return None
        return self._providers_by_ac[ac_name]

    def get_all_by_model(self, model=None):
        if model is None:
            return None
        if model not in self._providers_by_model:
            return None
        return self._providers_by_model[model]

    def get_provider_setting(self, provider, setting_name):
        """
        Get a provider specific setting.
        If that doesn't eixst, fall back to the global version of the setting.
        """
        # Provider specific version
        try:
            provider_settings = getattr(provider, 'settings')
            return provider_settings[setting_name]
        except (AttributeError, KeyError):
            return getattr(settings, setting_name)

    def set_provider_setting(self, provider, setting_name, setting_value):
        """
        Set a provider specific setting.
        Note: This is probably only be used by the test suite to test override settings
        post registration so we can assure setting overriding works
        """
        try:
            provider_settings = getattr(provider, 'settings')
        except AttributeError:
            setattr(provider, 'settings', {})
            provider.settings[setting_name] = setting_value

    def del_provider_setting(self, provider, setting_name):
        """
        Delete an provider specific setting.
        Note: This is probably only be used by the test suite to test override settings
        post registration so we can assure setting overriding works
        """
        try:
            provider_settings = getattr(provider, 'settings')
            del(provider_settings[setting_name])
        except (AttributeError, KeyError):
            return

    def get_ac_provider_setting(self, ac_name, provider, setting_name):
        """
        Get an autocompleter/provider specific setting.
        If that doesn't exist, fallback to the provider specific setting.
        If that doesn't eixst, fall back to the global version of the setting.
        """
        # AC/Provider specific version
        combined_name = "%s%s" % (ac_name, provider,)
        if setting_name in self._ac_provider_settings[combined_name]:
            return self._ac_provider_settings[combined_name][setting_name]
        # Provider specific version
        try:
            provider_settings = getattr(provider, 'settings')
            setting_value = provider_settings['setting_name']
            return setting_value
        except (KeyError, AttributeError):
            # Global version
            return getattr(settings, setting_name)

    def set_ac_provider_setting(self, ac_name, provider, setting_name, setting_value):
        """
        Set an autocompleter/provider specific setting.
        Note: This is probably only be used by the test suite to test override settings
        post registration so we can assure setting overriding works
        """
        combined_name = "%s%s" % (ac_name, provider,)
        self._ac_provider_settings[combined_name][setting_name] = setting_value

    def del_ac_provider_setting(self, ac_name, provider, setting_name):
        """
        Delete an autocompleter/provider specific setting.
        Note: This is probably only be used by the test suite to test override settings
        post registration so we can assure setting overriding works
        """
        combined_name = "%s%s" % (ac_name, provider,)
        if setting_name in self._ac_provider_settings[combined_name]:
            del self._ac_provider_settings[combined_name][setting_name]


registry = AutocompleterRegistry()


def add_obj_to_autocompleter(sender, instance, created, **kwargs):
    if instance is None:
        return

    providers = registry.get_all_by_model(sender)
    for provider in providers:
        provider(instance).store()


def remove_old_obj_from_autocompleter(sender, instance, **kwargs):
    try:
        old_instance = sender.objects.get(pk=instance.pk)

        providers = registry.get_all_by_model(sender)
        for provider in providers:
            provider(old_instance).remove()
    except sender.DoesNotExist:
        return


def remove_obj_from_autocompleter(sender, instance, **kwargs):
    if instance is None:
        return

    providers = registry.get_all_by_model(sender)
    for provider in providers:
        provider(instance).remove()


class AutocompleterSignalRegistry(object):
    def register(self, model):
        post_save.connect(add_obj_to_autocompleter, sender=model,
            dispatch_uid='autocompleter.%s.add' % (model))
        pre_save.connect(remove_old_obj_from_autocompleter, sender=model,
            dispatch_uid='autocompleter.%s.remoe_old' % (model))
        post_delete.connect(remove_obj_from_autocompleter,
            sender=model, dispatch_uid='autocompleter.%s.remove' % (model))

    def unregister(self, model):
        post_save.disconnect(add_obj_to_autocompleter,
            sender=model, dispatch_uid='autocompleter.%s.add' % (model))
        pre_save.disconnect(remove_old_obj_from_autocompleter, sender=model,
            dispatch_uid='autocompleter.%s.remoe_old' % (model))
        post_delete.disconnect(remove_obj_from_autocompleter,
            sender=model, dispatch_uid='autocompleter.%s.remove' % (model))

signal_registry = AutocompleterSignalRegistry()
