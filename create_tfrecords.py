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
from utils.dirs import data_dir, label_path
from reshape import reshape

def create_tf_example(data,
                      imagepath,
                      label_map_dict,
                      filename,
                      size,
                      ignore_difficult_instances=True
                      ):

  full_path = os.path.join(imagepath, filename + '.png')
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)

  resized_image = reshape(image, size)
  resized_encoded = tf.io.encode_png(resized_image)
  
  if image.format != 'PNG':
    raise ValueError('Image format not PNG')
  
  widthr, heightr = image.size
  #assert widthr == 512 and heightr == 512 , 'WRONG IMAGE SIZE'


  width = widthr
  height = heightr

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
     
    #print(obj['bndbox'])
    xmin = max(obj['bndbox'][0], 0)
    ymin = max(obj['bndbox'][1], 0)
    xmax = min(obj['bndbox'][2], width - 1)
    ymax = min(obj['bndbox'][3], height - 1)
    

    difficult_obj.append(int(difficult))

    xmins.append(float(xmin) / width)
    ymins.append(float(ymin) / height)
    xmaxs.append(float(xmax) / width)
    ymaxs.append(float(ymax) / height)

    obj['name'] = obj['name'].replace('-', ' ')
    classes_text.append(obj['name'].encode('utf8'))
    if (obj['name'] in label_map_dict):
        classes.append(label_map_dict[obj['name']])
    

    else:
        #print '>>>>>>>>>>>>>'
        continue

  assert len(xmins) == len(classes), 'wrong classes size'
  assert any([x < 0 for x in xmins]) == False, 'xmins'
  assert any([x < 0 for x in xmaxs]) == False, 'xmaxs'
  assert any([x < 0 for x in ymins]) == False, 'ymins'
  assert any([x < 0 for x in ymaxs]) == False, 'ymaxs'



  tf_example = tf.train.Example(features=tf.train.Features(feature={
      #'image/height': dataset_util.int64_feature(height),
      #'image/width': dataset_util.int64_feature(width),
      #'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      #'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(resized_encoded),
      #'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example


def create_records(data_dir, label_map_path, shard=256, shard_num=5, size=0,  images_dir='train'):

    if not os.path.exists(data_dir):
        raise ValueError('Error: no data_dir where {}'.format(data_dir))
    if not os.path.exists(data_dir):
        raise ValueError('Error: no label_map where {}'.format(label_map_path))

    output_path = os.path.join(data_dir, 'tf_records')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    imagepath = os.path.join(data_dir, 'images\\{}'.format(images_dir))
    labelpath = os.path.join(data_dir, 'labelTxt\\{}'.format(images_dir))   
    txtlist = [os.path.join(labelpath, x) for x in os.listdir(labelpath)]
    counter = 0

    for num in range(shard_num):

      writer = tf.io.TFRecordWriter(os.path.join(output_path, 'dota_r_{}{}{}_{}.tfrecords'.format(shard, images_dir, size, num+1)))
      #print ('start making record shard num {}'.format(num))
      
      
      #for fullname in txtlist:
      for i in range(counter, len(txtlist)):
        counter += 1
        if counter // (num + 1) == shard : break
        data = util.parse_bod_rec(txtlist[i])
        if len(data) == 0: 
          counter -= 1
          continue
        basename = os.path.basename(os.path.splitext(txtlist[i])[0])
        label_map_dict = label_map_util.get_label_map_dict(label_map_path)
        tf_example = create_tf_example(data,
                                       imagepath,
                                       label_map_dict,
                                       filename=basename,
                                       size=size,
                                       )
        writer.write(tf_example.SerializeToString())
      writer.close()



if __name__ == '__main__':
    create_records(data_dir, label_path, shard=64, shard_num=4, size=1024,  images_dir='train')
