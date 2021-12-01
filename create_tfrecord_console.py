import tensorflow as tf
import sys
import os
import io
import PIL.Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import xml.etree.ElementTree as ET
import codecs
import cv2
import numpy as np
import random
import shutil
import shapely.geometry as shgeo
import re
import pickle
import math
import copy
from absl import flags

def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

small_count = 0
def parse_bod_poly(filename):
    objects = []
    print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                #print 'splitline 9: ', splitlines[9]
                # if splitlines[9] == '1':
                if (splitlines[9] == 'tr'):
                    object_struct['difficult'] = '1'
                    #print '<<<<<<<<<<<<<<<<<<<<<<<<<<'
                else:
                    object_struct['difficult'] = splitlines[9]
                    #print '!!!!!!!!!!!!!!!!!!'
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            poly = list(map(lambda x:np.array(x), object_struct['poly']))
            object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            if (object_struct['long-axis'] < 15):
                object_struct['difficult'] = '1'
                global small_count
                small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects

def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                            max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax


def parse_bod_rec(filename):
    objects = parse_bod_poly(filename)
    for obj in objects:
        poly = obj['poly']
        bbox = dots4ToRec4(poly)
        obj['bndbox'] = bbox
    return objects

SAMPLES_PER_RECORD = 200



flags.DEFINE_string('data_dir', r'pathtodota', 'Root directory to raw bod dataset.')
flags.DEFINE_string('label_map_path', r'',
                    'Path to label map proto')
FLAGS = flags.FLAGS
def create_tf_example(data,
                      imagepath,
                      label_map_dict,
                      filename,
                      ignore_difficult_instances=True
                      ):
  # TODO(user): Populate the following variables from your example.

  full_path = os.path.join(imagepath, filename + '.jpg')
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  # if image.format != 'JPEG':
  #   raise ValueError('Image format not JPEG')

  #width = 1024
  #height = 1024
#   width = 608
#   height = 608
  width, height = image.size
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  difficult_obj = []
  for obj in data:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue
    # if ((float(obj['bndbox'][0]) < 0) or
    #     (float(obj['bndbox'][1]) < 0) or
    #     (float(obj['bndbox'][2]) >= 1024) or
    #     (float(obj['bndbox'][3]) >= 1024) ):
    #     continue
    xmin = max(obj['bndbox'][0], 0)
    ymin = max(obj['bndbox'][1], 0)
    xmax = min(obj['bndbox'][2], width - 1)
    ymax = min(obj['bndbox'][3], height - 1)

    difficult_obj.append(int(difficult))

    xmins.append(float(xmin) / width)
    ymins.append(float(ymin) / height)
    xmaxs.append(float(xmax) / width)
    ymaxs.append(float(ymax) / height)

    # xmins.append(float(obj['bndbox'][0]) / width)
    # ymins.append(float(obj['bndbox'][1]) / height)
    # xmaxs.append(float(obj['bndbox'][2]) / width)
    # ymaxs.append(float(obj['bndbox'][3]) / height)

    classes_text.append(obj['name'].encode('utf8'))
    if (obj['name'] in label_map_dict):
        classes.append(label_map_dict[obj['name']])

    else:
        #print '>>>>>>>>>>>>>'
        continue


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  #print 'tf_example: ', tf_example
  return tf_example


def get_output_filename(tf_records_name, fidx, file_num):
    return '%s.record-%05d-%05d' % (tf_records_name, fidx, file_num)


def tf_write(testortrain, tf_records_name):
    """
    :param testortrain: This is the index file for training data and test data. Please put them under the FLAGS.Data_dir path.
    :param tf_records_name:
    :return:
    """
    #writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'tf_records', tf_records_name))

    print ('start-------')
    # TODO(user): Write code to read in your dataset to examples variable
    data_dir = FLAGS.data_dir

    setname = os.path.join(data_dir, testortrain)
    imagepath = os.path.join(data_dir, 'JPEGImages')
    f = open(setname, 'r')
    lines = f.readlines()
    txtlist = [x.strip().replace(r'JPEGImages', r'wordlabel').replace('.jpg', '.txt') for x in lines]
    #txtlist = util.GetFileFromThisRootDir(os.path.join(data_dir, 'wordlabel'))
    i = 0
    fidx = 0
    file_num = len(txtlist)/SAMPLES_PER_RECORD + 1
    # for fullname, i in enumerate(txtlist):
    #     data = util.parse_bod_rec(fullname)
    #     #print 'len(data):', len(data)
    #     #print 'data:', data
    #     #assert len(data) >= 0, "there exists empty data: " + fullname
    #     basename = os.path.basename(os.path.splitext(fullname)[0])
    #     label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    #     #print 'label_map_dict', label_map_dict
    #     tf_example = create_tf_example(data,
    #                                    imagepath,
    #                                    label_map_dict,
    #                                    basename)
    #
    #     writer.write(tf_example.SerializeToString())
    # writer.close()
    while i < len(txtlist):
        # Open new TFRecord file.
        tf_filename = get_output_filename(tf_records_name, fidx, file_num)
        with tf.io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'tf_records', tf_filename)) as tfrecord_writer:
            j = 0
            while i < len(txtlist) and j < SAMPLES_PER_RECORD:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(txtlist)))
                sys.stdout.flush()

                fullname = txtlist[i]
                data = parse_bod_rec(fullname)
                basename = os.path.basename(os.path.splitext(fullname)[0])
                label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
                tf_example = create_tf_example(data,
                                               imagepath,
                                               label_map_dict,
                                               basename)
                tfrecord_writer.write(tf_example.SerializeToString())
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the DOTA dataset!')

def main(_):
    tf_write('val.txt', 'dota_val')
    tf_write('train.txt', 'dota_train')
    #tf_write('test.txt', 'dota_test_608.record')
    #tf_write('train.txt', 'dota_train_608.record')

if __name__ == '__main__':
  tf.compat.v1.app.run()