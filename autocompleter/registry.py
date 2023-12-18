from collections import OrderedDict

from django.db.models.signals import post_save, post_delete
from functools import partial

from autocompleter import settings


class AutocompleterRegistry(object):
    def __init__(self):
        self._ac_settings = OrderedDict()
        self._ac_provider_settings = OrderedDict()
        self._providers_by_ac = OrderedDict()
        self._providers_by_model = OrderedDict()

    def register(self, ac_name, provider, ac_provider_settings=None):
        """
        Register an autocompleter with a provider.
        Each autocompleter can have multiple providers.
        """
        if provider is None:
            return

        if ac_name not in self._providers_by_ac:
            self._providers_by_ac[ac_name] = []
            self._ac_settings[ac_name] = {}
        if getattr(provider, "model", None) is not None:
            if provider.model not in self._providers_by_model:
                self._providers_by_model[provider.model] = []
            if provider not in self._providers_by_model[provider.model]:
                self._providers_by_model[provider.model].append(provider)

        if provider not in self._providers_by_ac[ac_name]:
            self._providers_by_ac[ac_name].append(provider)

        combined_name = "%s%s" % (
            ac_name,
            provider,
        )
        # Note: the reason we default ac_provider_settings to None, then set to a dict is when we had
        # ac_provider_settings default to {} it was a reference to the same dict so when a setting
        # for one AC/provider was set, it was set for all AC/provider pairs.
        if ac_provider_settings is None:
            ac_provider_settings = {}
        self._ac_provider_settings[combined_name] = ac_provider_settings

    def unregister(self, ac_name, provider):
        """
        Unregister a provider from the autocompleter.
        """
        if provider is None:
            return
        if (
            ac_name in self._providers_by_ac
            and provider in self._providers_by_ac[ac_name]
        ):
            self._providers_by_ac[ac_name].remove(provider)
        if getattr(provider, "model", None) != None:
            if (
                provider.model in self._providers_by_model
                and provider in self._providers_by_model[provider.model]
            ):
                self._providers_by_model[provider.model].remove(provider)

        combined_name = "%s%s" % (
            ac_name,
            provider,
        )
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

    def get_autocompleter_setting(self, ac_name, setting_name):
        """
        Get an autocompleter specific setting.
        If that doesn't exist, fall back to the global version of the setting.
        """
        try:
            return self._ac_settings[ac_name][setting_name]
        except (AttributeError, KeyError):
            return getattr(settings, setting_name)

    def set_autocompleter_setting(self, ac_name, setting_name, setting_value):
        """
        Set an autocompleter specific setting.
        """
        self._ac_settings[ac_name][setting_name] = setting_value

    def del_autocompleter_setting(self, ac_name, setting_name):
        """
        Delete an autocompleter specific setting.
        """
        if setting_name in self._ac_settings[ac_name]:
            del self._ac_settings[ac_name][setting_name]

    def get_provider_setting(self, provider, setting_name):
        """
        Get a provider specific setting.
        If that doesn't exist, fall back to the global version of the setting.
        """
        # Provider specific version
        try:
            provider_settings = getattr(provider, "settings")
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
            provider_settings = getattr(provider, "settings")
        except AttributeError:
            setattr(provider, "settings", {})
            provider.settings[setting_name] = setting_value

    def del_provider_setting(self, provider, setting_name):
        """
        Delete an provider specific setting.
        Note: This is probably only be used by the test suite to test override settings
        post registration so we can assure setting overriding works
        """
        try:
            provider_settings = getattr(provider, "settings")
            del provider_settings[setting_name]
        except (AttributeError, KeyError):
            return

    def get_ac_provider_setting(self, ac_name, provider, setting_name):
        """
        Get an autocompleter/provider specific setting.
        If that doesn't exist, fallback to the provider specific setting.
        If that doesn't eixst, fall back to the global version of the setting.
        """
        # AC/Provider specific version
        combined_name = "%s%s" % (
            ac_name,
            provider,
        )
        if setting_name in self._ac_provider_settings[combined_name]:
            return self._ac_provider_settings[combined_name][setting_name]
        # Provider specific version
        try:
            provider_settings = getattr(provider, "settings")
            setting_value = provider_settings["setting_name"]
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
        combined_name = "%s%s" % (
            ac_name,
            provider,
        )
        self._ac_provider_settings[combined_name][setting_name] = setting_value

    def del_ac_provider_setting(self, ac_name, provider, setting_name):
        """
        Delete an autocompleter/provider specific setting.
        Note: This is probably only be used by the test suite to test override settings
        post registration so we can assure setting overriding works
        """
        combined_name = "%s%s" % (
            ac_name,
            provider,
        )
        if setting_name in self._ac_provider_settings[combined_name]:
            del self._ac_provider_settings[combined_name][setting_name]


registry = AutocompleterRegistry()


def add_obj_to_autocompleter(sender, instance, created, add_error_handler = None, remove_error_handler = None, **kwargs):
    if instance is None:
        return

    provider_classes = registry.get_all_by_model(sender)
    for provider_class in provider_classes:
        provider = provider_class(instance)
        if provider.include_item():
            try:
                provider.store()
            except Exception as e:
                if add_error_handler:
                    add_error_handler(instance, e)
                else:
                    raise e
        else:
            # If the item no longer passes the .include_item()
            # check then we need to remove it.
            try:
                provider.remove()
            except Exception as e:
                if remove_error_handler:
                    remove_error_handler(instance, e)
                else:
                    raise e


def remove_obj_from_autocompleter(sender, instance, remove_error_handler = None, **kwargs):
    if instance is None:
        return

    provider_classes = registry.get_all_by_model(sender)
    for provider_class in provider_classes:
        try:
            provider_class(instance).remove()
        except Exception as e:
            if remove_error_handler:
                remove_error_handler(instance, e)
            else:
                raise e

def remove_obj_from_autocompleter_with_error_handler(error_handler, sender, instance, **kwargs):
    return remove_obj_from_autocompleter(sender, instance, remove_error_handler=error_handler, **kwargs)

def add_obj_to_autocompleter_with_error_handler(add_error_handler, remove_error_handler, sender, instance, **kwargs):
    return add_obj_to_autocompleter(sender, instance, add_error_handler=add_error_handler, remove_error_handler=remove_error_handler, **kwargs)

class AutocompleterSignalRegistry(object):
    DISPATCH_ID_FUNCTION_MAPPING = {}

    def register(self, model, add_error_handler = None, remove_error_handler = None):

        add_uid = "autocompleter.%s.add" % (model)
        remove_uid = "autocompleter.%s.remove" % (model)

        if add_error_handler or remove_error_handler:
            remove_function = partial(remove_obj_from_autocompleter_with_error_handler, remove_error_handler)
            add_function = partial(add_obj_to_autocompleter_with_error_handler, add_error_handler, remove_error_handler)

            self.DISPATCH_ID_FUNCTION_MAPPING[add_uid] = add_function
            self.DISPATCH_ID_FUNCTION_MAPPING[remove_uid] = remove_function

        post_save.connect(
            self.DISPATCH_ID_FUNCTION_MAPPING.get(add_uid, add_obj_to_autocompleter),
            sender=model,
            dispatch_uid=add_uid,

        )
        post_delete.connect(
            self.DISPATCH_ID_FUNCTION_MAPPING.get(remove_uid, remove_obj_from_autocompleter),
            sender=model,
            dispatch_uid=remove_uid,
        )

    def unregister(self, model):
        add_uid = "autocompleter.%s.add" % (model)
        remove_uid = "autocompleter.%s.remove" % (model)
        post_save.disconnect(
            self.DISPATCH_ID_FUNCTION_MAPPING.get(add_uid, add_obj_to_autocompleter),
            sender=model,
            dispatch_uid=add_uid,
        )
        post_delete.disconnect(
            self.DISPATCH_ID_FUNCTION_MAPPING.get(remove_uid, remove_obj_from_autocompleter),
            sender=model,
            dispatch_uid=remove_uid,
        )

        self.DISPATCH_ID_FUNCTION_MAPPING.pop(add_uid, None)
        self.DISPATCH_ID_FUNCTION_MAPPING.pop(remove_uid, None)


signal_registry = AutocompleterSignalRegistry()
