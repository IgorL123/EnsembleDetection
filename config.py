import os
from object_detection.utils import config_util
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from utils.dirs import label_path,config_path, check_path, test_record_small_path, train_record_small_paths, out_path
from utils.dirs import robo_record_test,robo_record_train,robo_record_valid, robo_config

def config_update(label_path, config_path, check_path, test_record_path, train_record_paths, out_path):
 

    #config = config_util.get_configs_from_pipeline_file(config_path)
    labels = label_map_util.get_label_map_dict(label_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, config)
    
    config.model.ssd.num_classes = len(labels)
    config.train_config.batch_size = 2
    config.train_config.fine_tune_checkpoint = os.path.join(check_path)
    config.train_config.fine_tune_checkpoint_type = "detection"
    config.train_config.use_bfloat16 = False
    config.train_input_reader.label_map_path= label_path
    config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record_paths]
    config.eval_input_reader[0].label_map_path = label_path
    config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [test_record_path]
    #config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = 0.007


    config_text = text_format.MessageToString(config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(out_path, "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)

    print("\nSUCCESS\n")


if __name__ == '__main__':
    config_update(robo_config, config_path, check_path, robo_record_test, robo_record_valid, out_path)