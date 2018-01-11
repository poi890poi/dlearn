import tensorflow as tf, sys

NUM_PREDICTIONS = 5

MODELS = {
    'pre-trained' : {
        'graph' : '/usr/lib/cgi-bin/classify_image/imagenet/classify_image_graph_def.pb',
        'output' : 'softmax:0',
        'label' : '/usr/lib/cgi-bin/classify_image/imagenet/imagenet_labels_sorted.txt',
    },
    'flowers' : {
        'graph' : '/usr/lib/cgi-bin/classify_image/retrained/flowers/retrained_graph.pb',
        'output' : 'final_result:0',
        'label' : '/usr/lib/cgi-bin/classify_image/retrained/flowers/retrained_labels.txt',
    },
}

model = sys.argv[1]
SRC_GRAPH = MODELS[model]['graph']
SRC_LABEL = MODELS[model]['label']
TENSOR_OUTPUT = MODELS[model]['output']
SRC_IMAGE = sys.argv[2]

# Read in the image_data
image_data = tf.gfile.FastGFile(SRC_IMAGE, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(SRC_LABEL)]

# Unpersists graph from file
with tf.gfile.FastGFile(SRC_GRAPH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name(TENSOR_OUTPUT)
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-5:][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))