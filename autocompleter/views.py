import json

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseServerError
from django.views.generic import View

from autocompleter import settings
from autocompleter import Autocompleter


class SuggestView(View):
    def get(self, request, name):
        if settings.SUGGEST_PARAMETER_NAME in request.GET:
            term = request.GET[settings.SUGGEST_PARAMETER_NAME]
            ac = Autocompleter(name)
            if settings.FACET_PARAMETER_NAME in request.GET:
                facet_groups = request.GET[settings.FACET_PARAMETER_NAME]
                facet_groups = json.loads(facet_groups)
                if not self.validate_facets(facet_groups):
                    return HttpResponseBadRequest("Malformed facet parameter.")
                results = ac.suggest(term, facets=facet_groups)
            else:
                results = ac.suggest(term)

            json_response = json.dumps(results)
            return HttpResponse(json_response, content_type="application/json")
        return HttpResponseServerError("Search parameter not found.")

    @staticmethod
    def validate_facets(facet_groups):
        """
        Validates the list of facet groups has all the keys we expect as well
        as the correct facet types.
        """
        for facet_group in facet_groups:
            try:
                facet_type = facet_group["type"]
                if facet_type not in ["and", "or"]:
                    return False
                facet_dicts = facet_group["facets"]
                if len(facet_dicts) == 0:
                    return False
                for facet_dict in facet_dicts:
                    facet_dict["key"]
                    facet_dict["value"]
            except (KeyError, TypeError):
                return False
        return True


class ExactSuggestView(View):
    def get(self, request, name):
        if settings.SUGGEST_PARAMETER_NAME in request.GET:
            term = request.GET[settings.SUGGEST_PARAMETER_NAME]
            ac = Autocompleter(name)
            results = ac.exact_suggest(term)

            json_response = json.dumps(results)
            return HttpResponse(json_response, content_type="application/json")
        return HttpResponseServerError("Search parameter not found.")
