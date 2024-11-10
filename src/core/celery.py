from celery import Celery

celery_app = Celery(
    __name__,
    broker=str("redis://redis:6379/0"),
    backend=str("redis://redis:6379/0"),
)

celery_app.autodiscover_tasks([
    'src.celery_tasks.process_folder',
])
