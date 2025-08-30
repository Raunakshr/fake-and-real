"""WSGI config for Django project."""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_site.settings')

application = get_wsgi_application()
