import logging

from django.core.management.base import BaseCommand, CommandError

from autocompleter import registry


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

        input_provider_names = [name.strip() for name in options["autocompleter_providers"].split(",")]

        provider_name_to_provider = {
            provider.get_provider_name(): provider
            for providers in registry._providers_by_ac.values()
            for provider in providers
        }

        missing = [n for n in input_provider_names if n not in provider_name_to_provider]
        if missing:
            raise CommandError(f"No providers registered with these names: {missing}")

        provider_classes = [provider_name_to_provider[n] for n in input_provider_names]

        log_target = "autocompleter providers: %s" % ", ".join(input_provider_names)
 
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
