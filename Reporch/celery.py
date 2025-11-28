from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.signals import setup_logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Reporch.settings')

app = Celery('Reporch')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

from django.conf import settings
max_mb = int(getattr(settings, 'CELERY_MAX_MEM_PER_CHILD_MB', 0) or 0)
if max_mb > 0:
    app.conf.worker_max_memory_per_child = int(max_mb) * 1024
else:
    app.conf.worker_max_memory_per_child = 0
if getattr(app.conf, 'worker_prefetch_multiplier', None) is None:
    app.conf.worker_prefetch_multiplier = 2
if getattr(app.conf, 'worker_max_tasks_per_child', None) is None:
    app.conf.worker_max_tasks_per_child = 50

@setup_logging.connect
def configure_logging(*_args, **_kwargs):
    from logging.config import dictConfig
    dictConfig(settings.LOGGING)
