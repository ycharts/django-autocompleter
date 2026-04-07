from django.core.management import call_command
from django.test import TestCase


class ManagementCommandTestCase(TestCase):
    def test_reset_autocompleter_providers_callable(self):
        """
        Can call reset_autocompleter_providers without any error
        """
        try:
            call_command("reset_autocompleter_providers", autocompleter_providers="stock")
        except Exception:
            self.fail("Calling reset_autocompleter_providers has raised an exception unexpectedly")
