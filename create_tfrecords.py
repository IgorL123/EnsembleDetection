r"""Convert raw DOTA dataset to TFRecord for object_detection.
"""

import tensorflow as tf
import utils.utils as util
import sys
import os
import io
import PIL.Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


DATA_IMG_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\test'
DATA_LABEL_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\test'

DATA_IMG_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\train'
DATA_LABEL_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\train'

INDEX_FILE_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'
LABEL_MAP_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\label_map.pbtxt'
DATA_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'


def create_tf_example(data,
                      imagepath,
                      label_map_dict,
                      filename,
                      ignore_difficult_instances=True
                      ):

  full_path = os.path.join(imagepath, filename + '.png')
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)
  if image.format != 'PNG':
    raise ValueError('Image format not PNG')
  
  widthr, heightr = image.size

  # splitter = ImgSplitter(
  #           'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_toscale',
  #           'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled',
  #           subsize=800,
  #           gap=200,
  #           num_process=8,
  #       )
  
  # splitter.splitdata()

  width = 1024
  height = 1024

  xmins = [] 
  xmaxs = [] 

  ymins = [] 
  ymaxs = [] 

  classes_text = [] 
  classes = [] 
  difficult_obj = []
  for obj in data:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    xmin = max(obj['bndbox'][0], 0)
    ymin = max(obj['bndbox'][1], 0)
    xmax = min(obj['bndbox'][2], width - 1)
    ymax = min(obj['bndbox'][3], height - 1)

    difficult_obj.append(int(difficult))

    xmins.append(float(xmin) / width)
    ymins.append(float(ymin) / height)
    xmaxs.append(float(xmax) / width)
    ymaxs.append(float(ymax) / height)


    classes_text.append(obj['name'].encode('utf8'))
    if (obj['name'] in label_map_dict):
        classes.append(label_map_dict[obj['name']])


    else:
        #print '>>>>>>>>>>>>>'
        continue

  assert any([x < 0 for x in xmins]) == False, 'xmins'
  assert any([x < 0 for x in xmaxs]) == False, 'xmaxs'
  assert any([x < 0 for x in ymins]) == False, 'ymins'
  assert any([x < 0 for x in ymaxs]) == False, 'ymaxs'



  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example

def main():
    data_dir = DATA_DIR
    label_map_path = LABEL_MAP_PATH

    # if not os.path.exists(os.path.join(data_dir, indexfile)):
    #     # print os.path.join(data_dir, indexfile)
    #     raise ValueError('{} not in the path: {}'.format(indexfile, data_dir))

    output_path = os.path.join(data_dir, 'tf_records')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = tf.io.TFRecordWriter(os.path.join(output_path, 'dota_test2.tfrecords'))
    print ('start-------')

    imagepath = os.path.join(data_dir, 'images\\test')
    labelpath = os.path.join(data_dir, 'labelTxt\\test')
    txtlist = [os.path.join(labelpath, x) for x in os.listdir(labelpath)]

    for fullname in txtlist:
        data = util.parse_bod_rec(fullname)
        basename = os.path.basename(os.path.splitext(fullname)[0])
        label_map_dict = label_map_util.get_label_map_dict(label_map_path)
        tf_example = create_tf_example(data,
                                       imagepath,
                                       label_map_dict,
                                       basename)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
