# ========================================================================
# please check utils.py for all constants and imports used in this project
# ========================================================================
from utils import *


class ColorizeNet():
    """
    classification model part
    assertions:
        1. images should be converted to CIE L*a*b color space
        2. input images are grayscale (L component)
    """

    def __init__(self, alpha=(1/300), batch_size=26, h=256, w=256):
        self.batch_size = batch_size
        self.h = h
        self.w = w
        self.num_classes = NUM_CLASSES
        self.alpha = alpha
        self.lf_net = self.low_feature_net()
        self.gf_net = self.global_feature_net()
        self.model = self.build_color_net()
        self.trained = False
        self.labels = dict()
        self.labels_idx = dict()
        self.preprocessed = False

    def load_weights(self, weights_path, labels_map_path):
        """
        load model weights and class dictionary
        :param weights_path: path to trained weights
        """
        self.model.load_weights(weights_path)
        self.load_labels(labels_map_path)
        self.trained = True

    def load_labels(self, labels_path):
        """
        loads label map to class dictionary
        :param labels_path: path to label map file
        """
        counter = 0
        # load label map
        with open(labels_path, 'r') as f:
            for line in f:
                counter += 1
                tmp = line.split()
                assert len(tmp) == 2, "illegal file given"
                self.labels[tmp[0]] = int(tmp[1])
                self.labels_idx[int(tmp[1])] = tmp[0]
        self.num_classes = counter

    def preprocess(self, train_path, test_path, labels_path, gen_only_classes=False):
        """
        shuffles and splits data to: train, valid, test
        loads label map to self.labels
        """
        if gen_only_classes:
            list_just_classes()
        elif not os.path.exists(TRAIN_TXT):
            split_train_valid(train_path)
            shuffle_test(test_path)
        self.load_labels(labels_path)

    def load_dataset(self, dset_path):
        """
        :param dir_name: list of file names
        :return: generator that yields (gs_images_arr, [gt_color_images_arr, gt_classes_arr])
        """
        assert len(self.labels) > 0 or self.preprocessed is False, "please use preprocess method"

        # loop and yield batch
        counter = 0
        # inputs, ground_truth = [], []
        inputs, ground_truth_classes, ground_truth_imgs = [], [], []
        while True:
            with open(dset_path, 'r') as file:
                for line in file:
                    # read class and path
                    tmp = line.split()
                    if len(tmp) != 2:
                        continue
                    class_name, im_path = tmp[0], tmp[1]

                    # load image
                    try:
                        rgb = cv2.normalize(imread(im_path), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
                        lab = rgb2lab(rgb)
                        gs = lab[:, :, 0]
                    except Exception as e:  # ignores bad files/ paths
                        continue

                    # update lists
                    inputs.append(np.expand_dims(gs, -1))
                    z = np.zeros(self.num_classes)
                    z[self.labels[class_name]] = 1
                    ground_truth_classes.append(z)
                    ground_truth_imgs.append(lab)
                    counter += 1

                    # if batch is full -> send it
                    if counter == self.batch_size:
                        # yield inputs, ground_truth
                        yield np.array(inputs), [np.array(ground_truth_imgs), np.array(ground_truth_classes)]
                        inputs, ground_truth_classes, ground_truth_imgs = [], [], []
                        counter = 0

            # reached end of file -> reshuffle file
            reshuffle_file(dset_path, dset_path)

    def train(self, train_path, test_path, label_map_path, save_weights_dir,
              learning_rate=0.003, num_valid_steps=18000, steps_per_epoch=75000, num_epochs=10):

        # preprocess
        print("self.batch_size = {}".format(self.batch_size))
        print("self.alpha = {}".format(self.alpha))
        print("Begin Training:\n--Preprocess Phase--")
        if self.preprocessed is False:
            self.preprocess(train_path, test_path, label_map_path, True)
            self.preprocessed = True


        # init vars
        train_gen = self.load_dataset(TRAIN_TXT)
        valid_gen = self.load_dataset(VALID_TXT)

        # set checkpoint
        save_weights = ModelCheckpoint("{}/colorizeModelWeights".format(save_weights_dir),
                                       save_best_only=True, save_weights_only=True, verbose=1)  # /colorizeModelWeights.h5


        # set optimizer
        ada = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

        # if weights exists -> load them
        try:
            self.model.load_weights("{}/colorizeModelWeights".format(save_weights_dir))
        except Exception:
            print("no weights")

        # train model
        print("--Compile Model--")
        self.model.compile(optimizer="Adam", loss=["mean_squared_error", self.cross_ent_loss], loss_weights=[1, self.alpha])
        print("--Begin Fit--")
        self.model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                 epochs=num_epochs, validation_data=valid_gen, validation_steps=num_valid_steps,
                                 callbacks=[save_weights], verbose=1)


        for im in os.listdir(TRAIN_IM_DIR):
            img = cv2.normalize(imread("{}/{}".format(TRAIN_IM_DIR, im)), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            gs = rgb2lab(img)[:, :, 0].reshape((1, 256, 256, 1))
            img_p = self.model.predict(gs)[0]
            img_p = lab2rgb(img_p[0, :, :, :])
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_p)
            ax[0].set_title("predicted")
            ax[1].imshow(img)
            ax[1].set_title("ground truth")
            plt.show()
        self.trained = True

        # evaluate model
        print("--Evaluate Model--")
        test_gen = self.load_dataset(test_path)
        self.model.evaluate(test_gen)

    def predict_test(self, model_dir):

        # load model weights
        self.model.load_weights("{}/colorizeModelWeights".format(model_dir))

        for im in os.listdir(TRAIN_IM_DIR):
            img = cv2.normalize(imread("{}/{}".format(TRAIN_IM_DIR, im)), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            gs = rgb2lab(img)[:, :, 0].reshape((1, 256, 256, 1))
            img_p = self.model.predict(gs)[0]
            img_p = lab2rgb(img_p[0, :, :, :])
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_p)
            ax[0].set_title("predicted")
            ax[1].imshow(img)
            ax[1].set_title("ground truth")
            plt.show()

    def predict(self, input_img, classify=False):
        """
        predict a single image
        :param input_img:
        :param classify:
        :return:
        """
        # if not self.trained:
        #     print("uninitialized model, train model first")
        #     return
        if type(input_img) == str:
            im = cv2.cvtColor(plt.imread(input_img), cv2.COLOR_RGB2LAB)[:, :, 0]
        else:
            im = cv2.cvtColor(input_img, cv2.COLOR_RGB2LAB)[:, :, 0]
        if classify:
            return self.model.predict(im)[-1]
        return self.model.predict(im)[0]

    def low_feature_net(self):
        input_tensor = Input(shape=(None, None, 1))  # assuming grayscale input

        # low level features network shared layers
        llf_1 = Conv2D(filters=64, kernel_size=DEF_KERNEL, strides=(2, 2), padding=DEF_PAD, name="llf_1")
        llf_1a = BatchNormalization(name='bn_llf_1')
        llf_1b = Conv2D(filters=128, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD, name="llf_1b")
        llf_1ba = BatchNormalization(name='bn_llf_1b')
        llf_2 = Conv2D(filters=128, kernel_size=DEF_KERNEL, strides=(2, 2), padding=DEF_PAD, name="llf_2")
        llf_2a = BatchNormalization(name='bn_llf_2')
        llf_2b = Conv2D(filters=256, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD, name="llf_2b")
        llf_2ba = BatchNormalization(name='bn_llf_2b')
        llf_3 = Conv2D(filters=256, kernel_size=DEF_KERNEL, strides=(2, 2), padding=DEF_PAD, name="llf_3")
        llf_3a = BatchNormalization(name='bn_llf_3')
        llf_3b = Conv2D(filters=512, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD, name="llf_3b")
        llf_3ba = BatchNormalization(name='bn_llf_3a')

        # outputs of low level features
        llf_t1 = Activation(DEF_ACT)(llf_1a(llf_1(input_tensor)))
        llf_t1b = Activation(DEF_ACT)(llf_1ba(llf_1b(llf_t1)))
        llf_t2 = Activation(DEF_ACT)(llf_2a(llf_2(llf_t1b)))
        llf_t2b = Activation(DEF_ACT)(llf_2ba(llf_2b(llf_t2)))
        llf_t3 = Activation(DEF_ACT)(llf_3a(llf_3(llf_t2b)))
        llf_o1 = Activation(DEF_ACT, name="low_lvl_out")(llf_3ba(llf_3b(llf_t3)))

        return Model(inputs=[input_tensor], outputs=[llf_o1], name="low_feature_net")

    def global_feature_net(self):
        input_tensor = Input(shape=(28, 28, 512))  # assuming grayscale input

        gf_1 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_gf_1')
             (Conv2D(name='gf_1', filters=512, kernel_size=DEF_KERNEL, strides=(2, 2), padding=DEF_PAD)(input_tensor)))
        gf_1b = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_gf_1b')
             (Conv2D(name='gf_1b', filters=512, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(gf_1)))
        gf_2 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_gf_2')
             (Conv2D(name='gf_2', filters=512, kernel_size=DEF_KERNEL, strides=(2, 2), padding=DEF_PAD)(gf_1b)))
        gf_2b = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_gf_2b')
             (Conv2D(name='gf_2b', filters=512, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(gf_2)))
        gf_d = Activation(DEF_ACT)(BatchNormalization(name='bn_gf_d')(Dense(1024, name='gf_d')(tf.reshape(gf_2b, [-1, 1, 1, 7 * 7 * 512]))))
        global_feature_net_partial = Activation(DEF_ACT, name="gf_net_part")(BatchNormalization(name='bn_gf_p')(Dense(512, name='gf_partial')(gf_d)))

        return Model(inputs=[input_tensor], outputs=[global_feature_net_partial], name="global_feature_net")

    def build_classification_net(self):
        """
        builds the classification net part of the model
        :return:
        """
        input_tensor = Input(shape=(None, None, 1))  # assuming grayscale input

        # low level features net
        lf_net_out = self.lf_net(input_tensor)

        # global feature network
        global_feature_net_partial = self.gf_net(lf_net_out)

        # classification network
        cla_1 = Activation(DEF_ACT)(Dense(256, name='cla_1')(global_feature_net_partial))
        cla_out = tf.reshape(Dense(self.num_classes, name='cla_out_t')(cla_1), [-1, self.num_classes])
        cla_out = Softmax(axis=1, name="cla_out")(cla_out)

        return Model(inputs=[input_tensor], outputs=[cla_out], name="classification_model")

    def build_color_net(self):
        input_tensor = Input(shape=(self.h, self.w, 1))  # assuming grayscale input
        scaled_input = tf.image.resize(input_tensor, [224, 224], method="nearest")

        # global feature output
        llf_scaled = self.lf_net(scaled_input)
        global_feature_net_partial = self.gf_net(llf_scaled)

        # classification network
        cla_1 = Activation(DEF_ACT)(Dense(256, name='cla_1')(global_feature_net_partial))
        cla_out = tf.reshape(Dense(self.num_classes, name='cla_out_t')(cla_1), [-1, self.num_classes])
        cla_out = Softmax(axis=1, name="cla_out")(cla_out)

        # low-lvl feature output
        llf_out = self.lf_net(input_tensor)

        # mid level features
        mid_1 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_mid_1')
             (Conv2D(name='mid_1', filters=512, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(llf_out)))
        mid_2 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_mid_2')
             (Conv2D(name='mid_2', filters=256, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(mid_1)))

        # fusion layer
        global_feature_net_out = Activation(DEF_ACT)(BatchNormalization(name='bn_gf_out')
                                                     (Dense(256, name='gf_out')(global_feature_net_partial)))
        _, h, w, c = mid_2.shape
        tiled_glob = tfk.tile(global_feature_net_out, [1, h, w, 1])
        fused_layer = tf.concat([mid_2, tiled_glob], axis=-1)
        fusion_out = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_fusion')
             (Conv2D(name='fusion_out', filters=256, kernel_size=(1, 1), strides=(1, 1), padding=DEF_PAD)(fused_layer)))

        # colorization network
        col_1 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_col_1')
             (Conv2D(name='col_1', filters=128, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(fusion_out)))
        up_1 = UpSampling2D(name='up_1')(col_1)
        col_2 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_col_2')
             (Conv2D(name='col_2', filters=64, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(up_1)))
        col_2b = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_col_2b')
             (Conv2D(name='col_2b', filters=64, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(col_2)))
        up_2 = UpSampling2D(name='up_2')(col_2b)
        col_3 = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_col_3')
             (Conv2D(name='col_3', filters=32, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(up_2)))
        col_3b = Activation(DEF_ACT) \
            (BatchNormalization(name='bn_col_3b')
             (Conv2D(name='col_3b', filters=2, kernel_size=DEF_KERNEL, strides=(1, 1), padding=DEF_PAD)(col_3)))

        # color output
        resized_color = UpSampling2D(name='bn_col_resized')(col_3b)
        resized_color = tf.image.resize(resized_color, [self.h, self.w])
        col_out = tf.concat([input_tensor, resized_color], 3, name="color_out")

        # output the model
        model = Model(inputs=[input_tensor], outputs=[col_out, cla_out], name="colorization_model")
        model.summary()
        return model

    def frobinius_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.norm(y_pred-y_true, 'fro', axis=[1, 2], keepdims=False) ** 2)

    def cross_ent_loss(self, y_true, y_pred):
        y_class = tfk.flatten(tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=1))
        exponent = tf.keras.activations.exponential(y_pred)
        sum_exp = tfk.flatten(tf.math.reduce_sum(exponent, axis=1))
        log_term = tf.math.log(sum_exp)
        return -tf.math.reduce_mean(y_class - log_term)


