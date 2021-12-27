
import tensorflow as tf
from PIL import Image
import numpy as np
from six import BytesIO
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util



def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn




def set_paths(model):
  pipeline_config_path = 'D:\Programs\EnsembleDetection\EnsembleDetection\models\{}\pipeline.config'.format(model)
  model_dir = 'D:\Programs\EnsembleDetection\EnsembleDetection\models\{}'.format(model)
  checkpoint_dir = 'D:\Programs\EnsembleDetection\EnsembleDetection\models\{}\checkpoint\ckpt-0'.format(model)
  image_path = b'C:\Users\Zeden\Desktop\\P1398.png'
  label_map_path = b'C:\Users\Zeden\Desktop\\label_map.pbtxt'

  return pipeline_config_path, model_dir, checkpoint_dir, image_path, label_map_path



def get_prediction(model, save=False):
  pipeline_config_path, model_dir, checkpoint_dir, image_path, label_map_path = set_paths(model)


  configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
  model_config = configs['model']
  detection_model = model_builder.build(
        model_config=model_config, is_training=False)

  ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
  ckpt.restore(checkpoint_dir)


  label_map = label_map_util.load_labelmap(label_map_path)
  categories = label_map_util.convert_label_map_to_categories(
      label_map,
      max_num_classes=label_map_util.get_max_label_map_index(label_map),
      use_display_name=True)
  category_index = label_map_util.create_category_index(categories)


  detect_fn = get_model_detection_function(detection_model)
  image_np = load_image_into_numpy_array(image_path)

  input_tensor = tf.convert_to_tensor(
      np.expand_dims(image_np, 0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)

  label_id_offset = 1
  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
  )

  im = Image.fromarray(image_np_with_detections)
  if (save):
    im.save('res.png')
  else:
    return im


get_prediction('ssd_mobile_v2_5000', save=True)