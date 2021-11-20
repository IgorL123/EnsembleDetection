

import tensorflow as tf
from object_detection import model_lib_v2
import os


filenames = ['D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\\dota_train.tfrecords']
DATA_IMG_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\test'
DATA_LABEL_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\test'

DATA_IMG_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\train'
DATA_LABEL_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\train'

INDEX_FILE_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'
LABEL_MAP_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\LABELMAP.pbtxt'
DATA_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'
PIPELINE_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\pre-trained-models\efficientdet_d0_coco17_tpu-32\pipeline.config'


def main(unused_argv):

#   flags.mark_flag_as_required('model_dir')
#   flags.mark_flag_as_required('pipeline_config_path')
  tf.config.set_soft_device_placement(True)

  if FLAGS.checkpoint_dir:
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=FLAGS.checkpoint_dir,
        wait_interval=300, timeout=FLAGS.eval_timeout)
  else:
    if FLAGS.use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
      model_lib_v2.train_loop(
          pipeline_config_path=FLAGS.pipeline_config_path,
          model_dir=FLAGS.model_dir,
          train_steps=FLAGS.num_train_steps,
          use_tpu=FLAGS.use_tpu,
          checkpoint_every_n=FLAGS.checkpoint_every_n,
          record_summaries=FLAGS.record_summaries)

if __name__ == '__main__':
  main()