'''
This script parses classify_image label mapping from Inception V3 repository and save
sorted labels file that is compatible with image_retraining from Tensorflow repository.

classify_image from Inception V3 repository:
https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

image_retraining from Tensorflow repository:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining
'''

import tensorflow as tf
import os
import re

MODEL_DIR = './imagenet/'
      
label_lookup_path = os.path.join(
    MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
uid_lookup_path = os.path.join(
    MODEL_DIR, 'imagenet_synset_to_human_label_map.txt')

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
with open(os.path.join(MODEL_DIR, 'imagenet_labels_sorted.txt'), 'w') as f:
    f.write('\n') # Integer node ID of Inception V3 pre-trained model is one-based
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
        if val not in uid_to_human:
            tf.logging.fatal('Failed to locate: %s', val)
        name = uid_to_human[val]
        node_id_to_name[key] = name
        print(name)
        f.write(name + '\n')