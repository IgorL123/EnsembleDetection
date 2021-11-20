
from dotadevkit.ops import DataSplitter, ImgSplitter
#Not working 

splitter = ImgSplitter(
            'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_toscale',
            'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data_scaled',
            subsize=1024,
            gap=100,
            num_process=1,
        )
  

splitter.splitdata(1)