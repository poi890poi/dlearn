import os

print(os.environ['MODEL_DIR'])
print(os.path.join(os.environ['MODEL_DIR'], 'darknet/cfg/yolo.cfg'))
print(os.path.join('/usr/lib/cgi-bin/detection/models', 'darknet/darknet.weights'))
