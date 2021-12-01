import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from six import BytesIO
import matplotlib.pyplot as plt
from dirs import label_path, train_record_paths, test_path

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

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
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



category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
num_classes = 16
boxes_list, image_tensors, classes_list, shapes = get_data()

INDEX = 22
imgs_np = image_tensors[INDEX]

#img_np = imgs_np[INDEX]
boxes_np = np.array(boxes_list[INDEX], dtype=np.float32)
plt.figure(figsize=(30, 15))
plot_detections(
    imgs_np,
    boxes_np,
    classes_list[INDEX].numpy(),
    scores=None,
    category_index=category_index,
    image_name='res.png'
)

print(len(boxes_np))
print(shapes[INDEX])
print(classes_list[INDEX])

