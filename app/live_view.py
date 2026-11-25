"""Newest live image with recent detection overlays."""
import time
from app.detector import annotate_objects


def preview_frame(snapshot):
    # Always show the newest capture. Recent boxes expire instead of freezing video.
    if time.monotonic()-snapshot.reading_at<.25 and snapshot.objects:
        return annotate_objects(snapshot.frame,snapshot.objects)
    return snapshot.frame
