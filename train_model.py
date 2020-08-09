# ========================================================================
# please check utils.py for all constants and imports used in this project
# ========================================================================
from model import *


def train_model():
    """
    Trains a model.
    In case existing weights found in give weights path, it initiates the training with those weights
    :return: trained ColorizeNet model
    """
    if not os.path.exists(WEIGHTS_DIR):
        os.system("mkdir {}".format(WEIGHTS_DIR))
    # model = ColorizeNet(batch_size=2)
    model = ColorizeNet(alpha=1/300)
    model.train(TRAIN, TEST, CLASSES, WEIGHTS_DIR)
    return model


def predict_test(batch_size=18):
    """

    """
    # load model weights
    model = ColorizeNet(batch_size=batch_size)
    model.model.load_weights("{}_copy/colorizeModelWeights".format(WEIGHTS_DIR))

    model.load_labels("./classes.txt")
    test_gen = model.load_dataset(TEST)
    for counter in range(10):
        batch = next(test_gen)
        inputs, gt = batch[0], batch[1]
        pred = model.model.predict(inputs)[0]

        fig, ax = plt.subplots(3, 6, figsize=(30, 15))
        k = 0
        for i in range(batch_size):
            in_img = inputs[i,:, :, 0]
            ax[i//6][(k+0)%6].imshow(in_img, "gray")
            ax[i//6][(k+0)%6].set_title("input", fontsize=20)
            ax[i//6][(k+1)%6].imshow(lab2rgb(pred[i]))
            ax[i//6][(k+1)%6].set_title("prediction", fontsize=20)
            ax[i//6][(k+2)%6].imshow(lab2rgb(gt[0][i]))
            ax[i//6][(k+2)%6].set_title("ground_truth", fontsize=20)
            k += 3
        plt.show()


if __name__ == "__main__":

    # generate test train valid sets
    # gen_test_valid_train()

    # train model
    # model = train_model()

    # save trained model
    # if not os.path.exists(MODEL_DIR):
    #     os.system("mkdir {}".format(MODEL_DIR))
    # tf.keras.models.save_model(model.model, "{}".format(MODEL_DIR))

    # predict test
    predict_test()
