#!/usr/bin/env python3
import sys, os, socket
from socketserver import ThreadingMixIn
from http.server import SimpleHTTPRequestHandler, HTTPServer

from time import sleep
import time
timestamp = lambda: int(round(time.time() * 1000))

import http.server
import threading

import json
import uuid


HOST = socket.gethostname()


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SomethingSingleton(metaclass=Singleton):

    mId = ''

    def __init__(self):
        self.mId = str(uuid.uuid4())
        print('SomethingSingleton initialized ' + self.mId)
        print()

    def getuuid(self):
        self.mId = str(uuid.uuid4())
        return self.mId


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass


class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type',
                         'application/json;charset=utf-8')
        self.end_headers()

        now = -timestamp()

        http_response = {
            'thread' : {
                'id' : threading.get_ident(),
            },
            'timing' : {
                'elapsed' : now + timestamp(),
            },
            'something' : {
                'HOST' : HOST,
                'singleton' : SomethingSingleton().getuuid(),
            },
        }

        sleep(3)
        self.wfile.write(bytes(json.dumps(http_response), "utf8"))
        return


'''
This sets the listening port, default port 8080
'''
if sys.argv[1:]:
    PORT = int(sys.argv[1])
else:
    PORT = 8080

'''
This sets the working directory of the HTTPServer, defaults to directory where script is executed.
'''
if sys.argv[2:]:
    os.chdir(sys.argv[2])
    CWD = sys.argv[2]
else:
    CWD = os.getcwd()

server = ThreadingSimpleServer(('0.0.0.0', PORT), Handler)
print("Serving HTTP traffic from", CWD, "on", HOST, "using port", PORT)
try:
    while 1:
        sys.stdout.flush()
        server.handle_request()
except KeyboardInterrupt:
    print("\nShutting down server per users request.")