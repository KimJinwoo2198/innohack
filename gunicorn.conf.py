import multiprocessing
import os
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    worker_class = 'uvicorn.workers.UvicornWorker'
except ImportError:
    worker_class = 'uvicorn.workers.UvicornH11Worker'

workers = int(os.getenv('WEB_CONCURRENCY', str(multiprocessing.cpu_count() * 2 + 1)))
threads = int(os.getenv('WEB_THREADS', '2'))
bind = '0.0.0.0:8000'
timeout = int(os.getenv('WEB_TIMEOUT', '120'))
graceful_timeout = int(os.getenv('WEB_GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.getenv('WEB_KEEPALIVE', '5'))
preload_app = True
max_requests = 1000
max_requests_jitter = 50
loglevel = os.getenv('LOG_LEVEL', 'info')
accesslog = '-'
errorlog = '-'
forwarded_allow_ips = '*'