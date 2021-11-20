
import os

DATA_IMG_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\test'
DATA_LABEL_TEST_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\test'

DATA_IMG_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\images\\train'
DATA_LABEL_TRAIN_DIR = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data\labelTxt\\train'

INDEX_FILE_PATH = 'D:\ProgramData\TensorFlow\workspace\\training_det\dota\data'

def create_index_file(index_path, img_path):
    img_list = os.listdir(img_path)

    try:
        #os.chdir(index_path)
        with open(os.path.join(index_path, 'indexfile.txt'), 'w') as indexfile:
            for i in range(len(img_list)):
                img_str =  img_list[i][:-4] + '__1__0__{}'.format(i) + '.png'
                indexfile.write(os.path.join(img_path, img_str) + '\n')
        
            indexfile.close()
            
    except Exception:
        print(Exception)

    
    

if __name__ == '__main__':
    create_index_file(INDEX_FILE_PATH, DATA_IMG_TEST_DIR)
        