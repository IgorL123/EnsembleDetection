import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt


configs = config_util.get_configs_from_pipeline_file(b'C:\Users\Zeden\Desktop\center_res-bad\pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)


ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(b'C:\Users\Zeden\Desktop\center_res-bad\checkpoint', b'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
LABEL_MAP_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

img = cv2.imread('D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled512\images\\train\P0012__1.0__412___824.png')
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)


num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections


detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

im = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
cv2.imwrite('res.png', im)



