import os
import logging
import json
import time


from src.core.celery import celery_app


@celery_app.task
def process_tiff_prediction(details):
    time.sleep(10)
    print(details)
    return {"result": "success"}
