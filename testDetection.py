import os
import tensorflow as tf
import pathlib
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow._api.v2 import saved_model


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []

    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False)
        image_path =  pathlib.Path(image_path)
        image_paths.append(str(image_path))
    
    return image_paths


def load_labels():
    PATH_TO_LABELS = 'D:\ProgramData\TensorFlow\models\\research\object_detection\data\mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    # path = 'D:\\Programs\\EnsembleDet\\LABELMAP.pbtxt'
    # category_index = label_map_util.create_category_index_from_labelmap(path, use_display_name=True)

    return category_index


def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

def load_model(model_path='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'):
    base_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\pre-trained-models\\'
    path = base_path + model_path + '\saved_model'
    #path1 = 'D:\ProgramData\TensorFlow\workspace\\training_det\pre-trained-models\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model'
    path = pathlib.Path(path)#/"saved_model"
    model = tf.saved_model.load(str(path))
    print(model.signatures['serving_default'].inputs)
    print(model.signatures['serving_default'].output_dtypes)
    print(model.signatures['serving_default'].output_shapes)
    return model




def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    ts = tf.convert_to_tensor(image)
    input_tensor = ts[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    
    return output_dict



def show_inference(model, image_path):
    category_index = load_labels()
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
    im = Image.fromarray(image_np)
    im.save('res.png')
    #display(Image.fromarray(image_np))




PATH_TO_TEST_IMAGES_DIR = pathlib.Path('D:\ProgramData\TensorFlow\models\\research\object_detection\\test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
MODEL = 'efficientdet_d2_coco17_tpu-32'
detection_model = load_model(MODEL)


for image_path in TEST_IMAGE_PATHS:
    show_inference(detection_model, image_path)
