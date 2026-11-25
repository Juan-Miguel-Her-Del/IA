"""Continuous local JPEG preview in the Streamlit process, without page reruns."""
import atexit
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import weakref

import cv2

from app.live_view import preview_frame


class PreviewServer:
    def __init__(self):
        self.streams=weakref.WeakValueDictionary()
        self.frames=weakref.WeakKeyDictionary()
        self.lock=threading.Lock()
        registry=self

        class Handler(BaseHTTPRequestHandler):
            disable_nagle_algorithm=True
            def log_message(self,*args):pass

            def do_GET(self):
                token=self.path.removeprefix('/preview/') if self.path.startswith('/preview/') else ''
                with registry.lock:stream=registry.streams.get(token)
                if stream is None or not stream.alive:
                    self.send_error(404);return
                self.connection.settimeout(3.)
                self.send_response(200)
                self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control','no-store, no-cache, must-revalidate')
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers()
                number=-1
                try:
                    while stream.alive and not stream._stop.is_set():
                        snapshot,new_number=stream.wait_frame(number,timeout=.5)
                        if snapshot is None:continue
                        number=new_number
                        data=registry.jpeg(stream,snapshot,number)
                        if data is None:continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+
                                         str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n')
                        self.wfile.flush()
                except (OSError,TimeoutError):pass

        self.http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.http.daemon_threads=True
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True,name='vision-preview')
        self.thread.start()
        atexit.register(self.close)

    def url(self,stream):
        with self.lock:
            self.streams[stream.preview_token]=stream
            if stream not in self.frames:self.frames[stream]={'lock':threading.Lock(),'number':-1,'data':None}
        return f'http://127.0.0.1:{self.http.server_port}/preview/{stream.preview_token}'

    def jpeg(self,stream,snapshot,number):
        with self.lock:cache=self.frames[stream]
        # Reconnecting clients share one encoding of the newest frame.
        with cache['lock']:
            if number>cache['number']:
                rgb=preview_frame(snapshot)
                ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,75])
                if not ok:return None
                cache.update(number=number,data=jpeg.tobytes())
            return cache['data']

    def close(self):
        self.http.shutdown();self.http.server_close()
