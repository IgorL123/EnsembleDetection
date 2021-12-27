
import tensorflow as tf
from PIL import Image
import numpy as np
from six import BytesIO
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from ensemble_boxes import *
import os
from imagedet import get_model_detection_function, get_prediction, load_image_into_numpy_array


# Нужен еще функционал для оцени производительности
# В реальном вермени обнаружение тоже должно быть



class EnsembleDetection:
    """
    Create custom ensemble and predict bboxes

    models: list of strings - names of models from this models directory:
    center_net, center_net_bad, 
    d0, d0_512, d1, 
    d1_aug_bad, 
    ssd_mobile_v2, 
    ssd_mobile_v2_5000,
    (some).

    You can add your own object detection models in models directory 

    models_directory_path: path to directory with models
    
    """

    def __init__(self, models, models_directory_path, label_map_path, iou_thr=0.5):
        self.models_names = models
        self._models_path = models_directory_path
        self._label_map_path = label_map_path
        self.iou_thr = iou_thr
        self.number_of_models = len(models)
        
        self._SET_DET_MODELS()


    def _SET_DET_MODELS(self):
        dir = self._models_path
        mod = self.models_names

        DETECTION_MODELS = []
            
        for model in mod:
            
            main_path = os.path.join(dir, model)
            if not os.path.exists(main_path):
                raise ValueError('Error: no model dir where {}'.format(main_path))

            check_path = os.path.join(main_path,'checkpoint\ckpt-0')
            pipe_path = os.path.join(main_path, 'pipeline.config')

            if not os.path.exists(pipe_path):
                raise ValueError('Error: no pipeline config where {}'.format(pipe_path))


            configs = config_util.get_configs_from_pipeline_file(pipe_path)
            model_config = configs['model']
            detection_model = model_builder.build( model_config=model_config, is_training=False)
            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(check_path)


            det_func = get_model_detection_function(detection_model)
            DETECTION_MODELS.append(det_func)

        self.DET_MODELS = DETECTION_MODELS
        print("SETTING {} MODEL CONFIGS COMPLETED SUCCESSFULLY ".format(self.number_of_models))
    
    
    def _get_model_detection_function(model):
        """Get a tf.function for detection."""
        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn
    
    
    
    def predict(self, input_tensor):
        
        d_list = []
        s_list = []
        c_list = []

        for model in self.DET_MODELS:

            input_tensor_ex = tf.convert_to_tensor(np.expand_dims(input_tensor, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = model(input_tensor_ex)

            label_id_offset = 1
            image_np_with_detections = input_tensor.copy()
            image_np_with_detections_1 = input_tensor.copy()
            category_index = self._get_labels()
            classes_list= (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
            scores_list = detections['detection_scores'][0].numpy()
            detections_list = detections['detection_boxes'][0].numpy()

            d_list.append(detections_list)
            s_list.append(scores_list)
            c_list.append(classes_list)


    
        # boxes_1, scores_1, labels_1 = weighted_boxes_fusion(boxes_list = [d_list], 
        #                                 scores_list = [s_list],
        #                                 labels_list = [c_list], 
        #                                 weights = None, 
        #                                 iou_thr = self.iou_thr)

        boxes, scores, labels = nms(boxes = [d_list], 
                                                scores = [s_list],
                                                labels = [c_list], 
                                                weights = None, 
                                                iou_thr = self.iou_thr)
                    
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    boxes,
                    labels,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.5,
                    agnostic_mode=False,
                    )

        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #             image_np_with_detections_1,
        #             boxes_1,
        #             labels_1,
        #             scores_1,
        #             category_index,
        #             use_normalized_coordinates=True,
        #             max_boxes_to_draw=200,
        #             min_score_thresh=.5,
        #             agnostic_mode=False,
        #             )
            
            # im = Image.fromarray(image_np_with_detections)
            # if (save):
            #     im.save('res.png')
            # else:
            #     return im
        
        #return boxes, scores, labels
        return image_np_with_detections
        

                
    def _get_labels(self):
        if not os.path.exists(self._label_map_path):
                raise ValueError('Error: no label map where {}'.format(self._label_map_path))

        label_map = label_map_util.load_labelmap(self._label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        self.label_map = label_map
        return category_index
    
    def check_flops():
        pass



ens = EnsembleDetection(models=['ssd_mobile_v2_5000', 'd0_512'], 
                         models_directory_path='D:\Programs\EnsembleDetection\EnsembleDetection\models', 
                         label_map_path='D:\Programs\EnsembleDetection\EnsembleDetection\label_map.pbtxt')


image_np = load_image_into_numpy_array(b'C:\Users\Zeden\Desktop\\P0056.png')
res =  ens.predict(image_np)
im = Image.fromarray(res)
im.save('res1.png')

