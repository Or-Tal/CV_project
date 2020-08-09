import os
import numpy as np
import tensorflow.keras.backend as tfk
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D, Activation, BatchNormalization, Softmax
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint

#  =======  General Project Constants =======
CLASSIFICATION_INPUT_SHAPE = (224, 224, 1)
DEF_KERNEL = (3, 3)
FUSION_DEPTH = 256
EVAL_STEPS = 1e4
BATCH_SIZE = 25
NUM_CLASSES = 205
DEF_PAD = 'same'
DEF_ACT = 'relu'
SIG_ACT = 'sigmoid'
TRAIN_IM_DIR = "./in_train_im_test"
CLASS_WEIGHTS_DIR = "./weights_classification"
COL_WEIGHTS_DIR = "./weights_color"
TRAIN_TXT = "./train_shuffled.txt"
VALID_TXT = "./valid_shuffled.txt"
TEST_TXT = "./test_shuffled.txt"
DSET_PATH = "C:/images_dset"
TRAIN = "./train.txt"
TEST = "./test.txt"
CLASSES = "./classes.txt"
WEIGHTS_DIR = "./weights"
MODEL_DIR = "./models"
MODEL_NAME = "Colorization_model"
IMGS_DIR = "./images"
HANA_DIR = "./hana"
VALID_P = 0.2
TEST_P = 0.01
#  ==================================

def list_just_classes():
    # list dirs and subdirs
    subdirs = np.asarray(os.listdir(DSET_PATH)).flatten()  # directories: a - z

    # loop over all subdirs and classes -> write to txt file
    class_first = True
    counter = 0
    with open(CLASSES, 'w') as class_f:
        for subdir in subdirs:
            classes = os.listdir("{}/{}".format(DSET_PATH, subdir))
            for cla in classes:

                # add to class list
                s_class = "\n{} {}".format(cla, counter)
                if class_first:
                    s_class = s_class[1:]
                    class_first = False
                class_f.write(s_class)
                counter += 1


def gen_test_valid_train():
    """
    generates txt files from which the generators reads train and test data sets
    how to choose the validation set is left open for implementation
    """

    # list dirs and subdirs
    subdirs = np.asarray(os.listdir(DSET_PATH)).flatten()  # directories: a - z

    # loop over all subdirs and classes -> write to txt file
    test_first, train_first, class_first = True, True, True
    counter = 0
    with open(TEST, 'w') as test_f, open(TRAIN, 'w') as train_f, open(CLASSES, 'w') as class_f:
        for subdir in subdirs:
            classes = os.listdir("{}/{}".format(DSET_PATH, subdir))
            for cla in classes:

                # add to class list
                s_class = "\n{} {}".format(cla, counter)
                if class_first:
                    s_class = s_class[1:]
                    class_first = False
                class_f.write(s_class)
                counter += 1

                # print status
                print("class no. {}: {}".format(counter, cla))

                cur_path = "{}/{}/{}".format(DSET_PATH, subdir, cla)
                imgs = np.asarray(os.listdir(cur_path)).flatten()
                np.random.shuffle(imgs)
                threshold = int(np.size(imgs) * TEST_P)

                # add to relevant file by the format: <class> <img full path>
                for i, img in enumerate(imgs):
                    s = "\n{} {}/{}".format(cla, cur_path, img)
                    if i <= threshold:
                        if test_first:
                            s = s[1:]
                            test_first = False
                        test_f.write(s)
                    else:
                        if train_first:
                            s = s[1:]
                            train_first = False
                        train_f.write(s)


def shuffle_test(path):
    return reshuffle_file(path, TEST_TXT)


def split_train_valid(dset_path):
    # load all lines from file -> shuffle -> save train, valid set
    inputs = list()
    with open(dset_path, 'r') as f:
        for line in f:
            inputs.append(line)
    inputs = np.asarray(inputs).flatten()
    np.random.shuffle(inputs)
    N = inputs.shape[0]
    threshold = int(N * VALID_P)

    # write shuffled sets
    with open(TRAIN_TXT, 'w') as tmp_t, open(VALID_TXT, 'w') as tmp_v:
        for i, s in enumerate(inputs):
            if i <= threshold:
                if i < threshold:
                    s += "\n"
                tmp_v.write(s)
            else:
                if i < N - 1:
                    s += "\n"
                tmp_t.write(s)

    return N - threshold - 1, threshold + 1


def reshuffle_file(txt_file_path, wr_path):
    lines = []
    with open(txt_file_path, 'r') as f:
        for line in f:
            lines.append(line)

    # create a new shuffled file
    np.random.shuffle(lines)
    N = len(lines) - 1
    with open(wr_path, 'w') as f:
        for i, line in enumerate(lines):
            if i < N:
                line += '\n'
            f.write(line)

    return N + 1

