import logging

from django.core.management.base import BaseCommand, CommandError

from autocompleter import Autocompleter, registry


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--autocompleter_providers",
            action="store",
            dest="autocompleter_providers",
            required=True,
            help="Comma-separated names of autocompleter providers to initialize.",
            type=str,
        )
        parser.add_argument(
            "--remove",
            action="store_true",
            default=False,
            dest="remove",
            help="Remove all data for the given autocompleter providers.",
        )
        parser.add_argument(
            "--store",
            action="store_true",
            default=False,
            dest="store",
            help="Store all data for the given autocompleter providers.",
        )
        parser.add_argument(
            "--clear_cache",
            action="store_true",
            default=False,
            dest="clear_cache",
            help="Clear cache for autocompleter. Default to false.",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            default=False,
            dest="update",
            help="Updates all autocompleter data. Only processes objects that have been modified.",
        )

    help = "Reset autocompleter provider data"

    def handle(self, *args, **options):
        # Configure logging
        level = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}[
            options.get("verbosity", 0)
        ]
        logging.basicConfig(level=level, format="%(name)s: %(levelname)s: %(message)s")
        self.log = logging.getLogger("commands.reset_autocompleter_providers")

        provider_names = [name.strip() for name in options["autocompleter_providers"].split(",")]
        provider_classes = []
        for provider_name in provider_names:
            provider_class = None
            for ac_provider_classes in registry._providers_by_ac.values():
                for candidate_provider_class in ac_provider_classes:
                    if candidate_provider_class.get_provider_name() == provider_name:
                        provider_class = candidate_provider_class
                        break
                if provider_class is not None:
                    break
            if provider_class is None:
                raise CommandError(
                    "No provider named '%s' is registered in any autocompleter." % provider_name
                )
            provider_classes.append(provider_class)

        autocompleter_names = set()
        for pc in provider_classes:
            for ac_name, ac_provider_classes in registry._providers_by_ac.items():
                if pc in ac_provider_classes:
                    autocompleter_names.add(ac_name)
        autocompleters = [Autocompleter(name) for name in sorted(autocompleter_names)]

        log_target = "autocompleter providers: %s" % ", ".join(provider_names)
 
        if options["remove"]:
            self.log.info("Removing all objects for %s" % log_target)
            for pc in provider_classes:
                pc.remove_all()

        if options["store"]:
            self.log.info("Storing all objects for %s" % log_target)
            for pc in provider_classes:
                pc.store_all()

        if options["update"]:
            self.log.info("Updating all objects with updates for %s" % log_target)
            for pc in provider_classes:
                pc.update_all()

        if options["clear_cache"]:
            self.log.info("Clearing cache for %s" % log_target)
            for autocomp in autocompleters:
                autocomp.clear_cache()
