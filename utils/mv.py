import os,shutil
from random import shuffle
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def move_file(dirname):
    '''
    move npy file to training dir and test dir
    '''
    train_dir = os.path.join(BASE_PATH,'data/trainData/')
    test_dir = os.path.join(BASE_PATH,'data/testData/')
    if os.path.isdir(dirname):
        dirs = os.listdir(dirname)
        for directory in dirs:
            num = 0
            directoryName = os.path.join(dirname,directory)
            if os.path.isdir(directoryName):
                files = os.listdir(directoryName)
                shuffle(files)
                for file in files:
                    if file.endswith('.npy'):
                        filename = os.path.join(directoryName,file)
                        if num < 460:
                            shutil.move(filename,train_dir)
                            num = num + 1
                        else:
                            shutil.move(filename,test_dir)
                            num = num + 1