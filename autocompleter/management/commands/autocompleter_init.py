import logging

from django.core.management.base import BaseCommand, CommandError

from autocompleter import Autocompleter, registry


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--autocompleter_provider",
            action="store",
            dest="autocompleter_provider",
            required=True,
            help="Name of autocompleter provider to initialize.",
            type=str,
        )
        parser.add_argument(
            "--remove",
            action="store_true",
            default=False,
            dest="remove",
            help="Remove all autocompleter data. Default to false.",
        )
        parser.add_argument(
            "--store",
            action="store_true",
            default=False,
            dest="store",
            help="Store all autocompleter data. Default to false.",
        )
        parser.add_argument(
            "--clear_cache",
            action="store_true",
            default=False,
            dest="clear_cache",
            help="Clear cache for autocompleter. Default to false.",
        )
        parser.add_argument(
            "--skip_delete_old",
            action="store_false",
            default=True,
            dest="delete_old",
            help="Do not clear old terms from autocompleter when storing. "
            "Recommended only to be used with store all after remove_all otherwise orphan keys will remain.",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            default=False,
            dest="update",
            help="Updates all autocompleter data. Only processes objects that have been modified. "
            "Equivalent to --remove --clear_cache --store",
        )

    help = "Store and/or remove autocompleter data"

    def handle(self, *args, **options):
        # Configure logging
        level = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}[
            options.get("verbosity", 0)
        ]
        logging.basicConfig(level=level, format="%(name)s: %(levelname)s: %(message)s")
        self.log = logging.getLogger("commands.autocompleter_init")

        provider_name = options["autocompleter_provider"]
        provider_class = None
        for provider_classes in registry._providers_by_ac.values():
            for candidate_provider_class in provider_classes:
                if candidate_provider_class.get_provider_name() == provider_name:
                    provider_class = candidate_provider_class
                    break
            if provider_class is not None:
                break

        if provider_class is None:
            raise CommandError(
                "No provider named '%s' is registered in any autocompleter." % provider_name
            )

        should_clear_cache = options["remove"] or options["clear_cache"] or options["update"]
        autocompleters = []
        if should_clear_cache:
            # Cache keys are tied to autocompleter names, so to keep the same behavior
            # where .clear_cache() is called in remove_all or update_all we need to invalidate
            # every autocompleter cache related to this autocompleter provider. 
            autocompleter_names = [
                autocompleter_name
                for autocompleter_name, provider_classes in registry._providers_by_ac.items()
                if provider_class in provider_classes
            ]
            autocompleters = [
                Autocompleter(autocompleter_name)
                for autocompleter_name in sorted(autocompleter_names)
            ]

        log_target = "autocompleter provider: %s" % (provider_name)

        if options["remove"]:
            self.log.info("Removing all objects for %s" % (log_target))
            for obj in provider_class.get_iterator():
                provider_class(obj).remove()
            for autocomp in autocompleters:
                autocomp.clear_cache()

        if options["store"]:
            delete_old = options["delete_old"]
            self.log.info("Storing all objects for %s" % (log_target))
            for obj in provider_class.get_iterator():
                provider = provider_class(obj)
                if provider.include_item():
                    provider.store(delete_old=delete_old)

        if options["clear_cache"]:
            self.log.info("Clearing cache for %s" % (log_target))
            for autocomp in autocompleters:
                autocomp.clear_cache()

        if options["update"]:
            self.log.info("Updating all objects with updates for %s" % (log_target))
            # Update this provider once; then clear caches for all related autocompleters.
            updater = autocompleters[0]
            updater.update_provider(provider_class)
            for autocomp in autocompleters:
                autocomp.clear_cache()
