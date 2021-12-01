
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image
import numpy as np
from object_detection.utils import visualization_utils
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder
from object_detection.utils import label_map_util, dataset_util
from utils.dirs import data_dir, test_path, train_record_paths, label_path
import matplotlib.pyplot as plt

# TRY TO RESHAPE THE IMAGE


image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

def get_data():
    
    
    #lb_dict = label_map_util.get_label_map_dict(label_path)
    raw_dataset = tf.data.TFRecordDataset(train_record_paths[:1])
    raw_dataset = tf.data.TFRecordDataset(test_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    image_tensors = [ tf.io.decode_image(ex["image/encoded"]).numpy() for ex in parsed_dataset ]
    boxes_list = []
    classes_list = []
    shapes = []


    for ex in parsed_dataset:
        gt_boxes_list = []
        h = ex['image/height']
        w = ex['image/width']
        xmin = ex['image/object/bbox/xmin'].numpy() 
        xmax = ex['image/object/bbox/xmax'].numpy() 
        ymin = ex['image/object/bbox/ymin'].numpy() 
        ymax = ex['image/object/bbox/ymax'].numpy() 
        clasess= ex['image/object/class/text']
        label = ex['image/object/class/label']
        for i in range(len(xmin)):
            one_box = [ymin[i] / h ,xmin[i] / w, ymax[i] /h ,  xmax[i] / w ] # DO NOT FUCKING CHANGE THIS
            gt_boxes_list.append(one_box)
            classes_list.append(label)
            shapes.append([w ,h])

        boxes_list.append(gt_boxes_list)


    return boxes_list, image_tensors, classes_list, shapes 

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):

  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.5,
      groundtruth_box_visualization_color='red',
      )
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)



if __name__ == '__main__':
    boxes_list, image_tensors, classes_list, shapes = get_data()

    INDEX = 19


    img_np = image_tensors[INDEX]
    boxes_np = np.array(boxes_list[INDEX], dtype=np.float32)
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    # cv2.imshow("img",img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    resized_down = cv2.resize(img_np, (1024,1024),interpolation= cv2.INTER_LINEAR)
    # cv2.imshow('resized', resized_down)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    resized_down2 = cv2.resize(img_np, (640,640),interpolation= cv2.INTER_LINEAR)
    plt.figure(figsize=(30, 15))
    plot_detections(
        img_np,
        boxes_np,
        classes_list[INDEX].numpy(),
        scores=None,
        category_index=category_index,
        image_name='res1.png'
    )

    plt.figure(figsize=(30, 15))
    plot_detections(
        resized_down,
        boxes_np,
        classes_list[INDEX].numpy(),
        scores=None,
        category_index=category_index,
        image_name='res2.png'
    )

    plt.figure(figsize=(30, 15))
    plot_detections(
        resized_down2,
        boxes_np,
        classes_list[INDEX].numpy(),
        scores=None,
        category_index=category_index,
        image_name='res3.png'
    )


def reshape(img_np, size):
    resized = cv2.resize(img_np, (size,size), interpolation=cv2.INTER_LINEAR)
    return resized


