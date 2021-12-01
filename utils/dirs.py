
'''

IMPORTANT DIRECTORIES

'''


label_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\label_map.pbtxt'
test_record_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled1024\\tf_records\dota_100test1024.tfrecords'
train_record_paths_1024 = [
        'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled1024\\tf_records\dota_256train1024_0.tfrecords',
        'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled1024\\tf_records\dota_256train1024_1.tfrecords',
        'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled1024\\tf_records\dota_256train1024_2.tfrecords',]
config_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\pre-trained-models\efficientdet_d0_coco17_tpu-32\pipeline.config'
out_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\models\efficientdet_d0\pipeline.config'
check_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\pre-trained-models\efficientdet_d0_coco17_tpu-32\checkpoint\ckpt-0'


DATA_IMG_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\test'
DATA_LABEL_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\test'

DATA_IMG_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\train'
DATA_LABEL_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\train'

INDEX_FILE_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'
LABEL_MAP_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\label_map.pbtxt'
data_dir = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'


train_record_paths = [
        'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_0.tfrecords',
        'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_1.tfrecords',
        'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_2.tfrecords',]

test_path = ['D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_128train0_1.tfrecords']


train_record_small_paths = ['D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_1.tfrecords',
                            'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_2.tfrecords',
                            'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_3.tfrecords',]
test_record_small_path = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\\tf_records\dota_64train1024_4.tfrecords'




robo_record_train = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data2\\train\\no.tfrecord'
robo_record_valid = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data2\\valid\\no.tfrecord'
robo_record_test = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data2\\test\\no.tfrecord'
robo_config = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data2\\label_map.pbtxt'