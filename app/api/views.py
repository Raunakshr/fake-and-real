from django.http import JsonResponse


def predict(request):
    # TODO: Implement prediction logic
    data = {'label': 'TODO', 'proba': 0.0}
    return JsonResponse(data)
