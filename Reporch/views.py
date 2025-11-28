from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET

@require_GET
def health(request):
    """
    Health check endpoint.
    Returns HTTP 200 with status 'ok' if the service is running.
    """
    return JsonResponse({"status": "ok"}, status=200)

def empty_404(request, exception):
    return HttpResponse(status=404) 