from object_detection.utils import label_map_util
from tensorflow.python.util import compat
dir = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\label_map.pbtxt'


from tensorflow.python.lib.io import _pywrap_file_io
print(compat.path_to_str(dir))
_read_buf=_pywrap_file_io.BufferedInputStream(compat.path_to_str(dir), 1024*512)