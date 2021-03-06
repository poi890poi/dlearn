from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re

import numpy as np
from six.moves import urllib
import tensorflow as tf

import sys

import http.server
import socketserver
import urllib.request

import json

TOP_PREDICTIONS = 5
CLASSIFY_MAXLEN = 1058476
MODEL_DIR = './imagenet'

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          MODEL_DIR, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_data):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    plist = []

    top_k = predictions.argsort()[-TOP_PREDICTIONS:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      plist.append( '%s (score = %.5f)' % (human_string, score) )

    return plist


PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type',
                         'application/json;charset=utf-8')
        self.end_headers()

        length = int(self.headers['Content-Length'])
        dtype = self.headers['Arobot-Data-Type']
        rid = self.headers['Arobot-Request-Id']
        json_response = {
            'header' : {
                'api' : 'image_classify',
                'date_changed' : '2018-01-10',
                'request_id' : rid,
                'request_data_type' : dtype,
                'err_no' : 500,
                'err_msg' : 'Unexpected server error'
            },
            'image_classify' : {
                'model' : 'inception v3 pre-trained',
                'transfer_learn' : 'none',
            },
        }
        if length > CLASSIFY_MAXLEN:
          json_response['header']['err_no'] = 400
          json_response['header']['err_msg'] = 'Image data too large (max '+str(CLASSIFY_MAXLEN)+' bytes)'
          return

        if dtype=='url':
          url = self.rfile.read(length).decode('utf-8')
          with urllib.request.urlopen(url) as img:
            img_data = img.read()
            #image_data = tf.gfile.FastGFile(url, 'rb').read()
            json_response['image_classify']['url'] = url
            json_response['image_classify']['predicts'] = run_inference_on_image(img_data)
            json_response['header']['err_no'] = 0
            json_response['header']['err_msg'] = 'Success'
        elif dtype=='bin':
          json_response['image_classify']['predicts'] = run_inference_on_image(self.rfile.read(length))
          json_response['header']['err_no'] = 0
          json_response['header']['err_msg'] = 'Success'
          pass
        else:
          json_response['header']['err_no'] = 400
          json_response['header']['err_msg'] = 'Unsupported data type'
          pass

        self.wfile.write(bytes(json.dumps(json_response), "utf8"))

    def do_GET(self):
        # Construct a server response.
        self.send_response(200)
        self.send_header('Content-type',
                         'application/json;charset=utf-8')
        self.end_headers()

        json_response = {
            'header' : {
                'api' : 'image_classify',
                'date' : '2018-01-08',
                'err_no' : 500,
                'err_msg' : 'Unexpected server error'
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
        inpath = os.path.join( '/var/www/html/images/', os.path.basename(self.path) )
        json_response['image_classify']['input_file'] = inpath
        if os.path.exists(inpath):
            json_response['image_classify']['predicts'] = run_inference_on_image(inpath)
            json_response['header']['err_no'] = 0
            json_response['header']['err_msg'] = 'Success'
        else:
            json_response['header']['err_no'] = 404
            json_response['header']['err_msg'] = 'File not found'
        self.wfile.write(bytes(json.dumps(json_response), "utf8"))
        return

# Creates graph from saved GraphDef.
create_graph()

socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(("", PORT), Handler)
print("serving at port", PORT)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("releasing port", PORT)
