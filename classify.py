#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging

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

from hashlib import md5
import tempfile
import shutil
import os
import subprocess
import json

import time
timestamp = lambda: int(round(time.time() * 1000))

import cv2

import cgitb
import cgi
cgitb.enable()

CLASSIFY_DIMENSION = 299
TOP_PREDICTIONS = 5
MODEL_DIR = '/tmp/imagenet'

def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()

def classify():
    
    fs = cgi.FieldStorage()
    
    # Validate data
    json_response['input']['fs_type'] = fs.type
    json_response['input']['fs_headers'] = fs.headers
    if 'userfile' not in fs: return 0
    len = int(fs.headers['content-length'])
    if len > 4*1048576: return 0
    fileitem = fs['userfile']
    if not fileitem.file: return 0
    
    # Load as opencv image object
    data = fileitem.file.read()
    file_bytes = np.asarray( bytearray( data ), dtype=np.uint8 )
    img = cv2.imdecode( file_bytes, 1 )
    fname = md5digest(data) + '.jpg'
    outpath = os.path.join( '/var/www/html/images/', fname )
    
    # Resize and pad image
    height, width, channels = img.shape
    json_response['input']['dim_original'] = (width, height, channels)
    if width >= height: # Landscape
        height = int(CLASSIFY_DIMENSION * height / width)
        width = CLASSIFY_DIMENSION
    else:
        width = int(CLASSIFY_DIMENSION * width / height)
        height = CLASSIFY_DIMENSION
    json_response['input']['dim_transformed'] = (width, height, channels)
    resized_image = cv2.resize(img, (width, height))
    if width < CLASSIFY_DIMENSION:
        bleft = int((CLASSIFY_DIMENSION-width)/2)
        bright = CLASSIFY_DIMENSION - bleft - width
        resized_image = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=bleft, right=bright, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    elif height < CLASSIFY_DIMENSION:
        btop = int((CLASSIFY_DIMENSION-height)/2)
        bbottom = CLASSIFY_DIMENSION - btop - height
        resized_image = cv2.copyMakeBorder(resized_image, top=btop, bottom=bbottom, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    outpath = outpath+'.jpg'
    cv2.imwrite(outpath, resized_image)

    run_inference_on_image(outpath)


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


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  json_response['profiling']['create_graph'] = -timestamp()
  create_graph()
  json_response['profiling']['create_graph'] += timestamp()

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
    json_response['profiling']['sess.run'] = -timestamp()
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    json_response['profiling']['sess.run'] += timestamp()
    json_response['profiling']['squeeze'] = -timestamp()
    predictions = np.squeeze(predictions)
    json_response['profiling']['squeeze'] += timestamp()

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-TOP_PREDICTIONS:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      json_response['predictions'].append( '%s (score = %.5f)' % (human_string, score) )


print("Content-Type: application/json;charset=utf-8")
print() # End of HTTP response headers

json_response = {
    'env' : {},
    'input' : {},
    'profiling' : {},
    'predictions' : [],
}
json_response['env']['encoding'] = sys.stdout.encoding
json_response['env']['tensorflow'] = tf.__version__
json_response['env']['opencv'] = cv2.__version__

json_response['profiling']['total'] = -timestamp()

classify()

json_response['profiling']['total'] += timestamp()
print( json.dumps(json_response) )

#exec(open("../classify_image.py").read())