import http.server
import socketserver

import json

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        # Construct a server response.
        self.send_response(200)
        self.send_header('Content-type',
                         'application/json;charset=utf-8')
        self.end_headers()

        http_response = {
            'header' : {
                'api' : 'image_classify',
                'date' : '2018-01-08'
            },
            'image_classify' : {
                'model' : 'inception v3 pre-trained',
                'transfer_learn' : 'none',
            },
            'http' : {
                'method' : 'GET',
                'path' : self.path
            },
        }
        self.wfile.write(bytes(json.dumps(http_response), "utf8"))
        return

httpd = socketserver.TCPServer(("", PORT), Handler)
print("serving at port", PORT)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
	pass