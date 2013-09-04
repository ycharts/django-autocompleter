from optparse import make_option
import logging

from django.core.management.base import BaseCommand

import autocompleter
from autocompleter import Autocompleter

# Must autodiscover existing autocompleter providers before can initialize
autocompleter.autodiscover()


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option("--name", dest="name",
            help="Name of autocompleter to initialize. Defaults to default autocompleter name.",
            action="store", type="string"),
        make_option("--remove", dest="remove",
            help="Remove all autocompleter data. Default to false.",
            action="store_true", default=False),
        make_option("--store", dest="store",
            help="Store all autocompleter data. Default to false.",
            action="store_true", default=False),
        make_option("--clear_cache", dest="clear_cache",
            help="Clear cache for autocompleter. Default to false.",
            action="store_true", default=False),
        make_option("--skip_delete_old", dest="delete_old",
            help="Clear old terms from autocompleter.\
            Recommended only to be used with store all after remove_all otherwise \
            orphan keys will remain.",
            action="store_false", default=True),
    )

    help = "Store and/or remove autocompleter data"

    def handle(self, *args, **options):
        # Configure loggingin
        level = {
            '0': logging.WARN,
            '1': logging.INFO,
            '2': logging.DEBUG
        }[options.get('verbosity', '0')]
        logging.basicConfig(level=level, format="%(name)s: %(levelname)s: %(message)s")
        self.log = logging.getLogger('commands.autocompleter_init')

        autocomp = Autocompleter(options["name"])
        if options['remove']:
            self.log.info("Removing all objects for autocompleter: %s" % (options['name']))
            autocomp.remove_all()
        if options['store']:
            delete_old = options['delete_old']
            self.log.info("Storing all objects for autocompleter: %s" % (options['name']))
            autocomp.store_all(delete_old=delete_old)
        if options['clear_cache']:
            self.log.info("Clearing cache for autocompleter: %s" % (options['name']))
            autocomp.clear_cache()

