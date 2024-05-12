from ctypes.wintypes import RGB
import os
import shutil

logger_path = './logger'

def log_text(message, filename):
    logger = open(logger_path + "/" + filename, "a+")
    logger.write(message)
    logger.close()

def create_directory(path, directory_name=None):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

    try:
        os.mkdir(path)
    except OSError:
        if path == logger_path:
            print('Creation of the '+ str(directory_name) + ' ' + str(path) + 'failed')
        else:
            log_text('Creation of the '+ str(directory_name) + ' ' + str(path) + 'failed\n', str(directory_name) + ".txt")
    else:
        if path == logger_path:
            print('Successfully created the ' + str(directory_name) + ' ' + str(path))
        else:
            log_text('Successfully created the ' + str(directory_name) + ' ' + str(path) + '\n', str(directory_name) + ".txt")

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def createFolderFromPaths(paths):
    for path in paths:
        create_dir(path)

def removeFilesFromPaths(paths):
    for path in paths:
        path_dir = os.listdir(path)
        for file in path_dir:
            try:
                shutil.rmtree(path + file)
            except OSError:
                os.remove(path + file)

def removeFile(filepath):
    try:
        os.remove(filepath)
    except OSError:
        pass

alpha = 255
colors = {
    'RED':      (0,0,255,alpha),
    'ORANGE':   (8,147,253,alpha),
    'YELLOW':   (0,255,255,alpha),
    'GREEN':    (0,255,0,alpha),
    'BLUE':     (255,0,0,alpha),
    'VIOLET':   (128,0,128,alpha),
    'WHITE':    (255,255,255,alpha),
    'TEAL':     (255,255,127,alpha)
}