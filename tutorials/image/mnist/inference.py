import tensorflow as tf
import numpy as np
import cv2
import time
import glob
import os
import shutil

FLAGS = None
NUM_CHANNELS = 1
IMAGE_SIZE = 28
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.

_DEBUG_ = True

def save_debug_img(is_debug, img_path, img):
    if is_debug:
        cv2.imwrite(img_path, img)

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32

def get_square_image(input_image, dst_size):
    """
    make image to squared, and resize to dst size
    :param input_image:
    :param dst_size:
    :return: resized image
    """
    height,width = input_image.shape
    if(height > width):
        borderWith = (height - width)/2
        input_image = cv2.copyMakeBorder(input_image, 0, 0, borderWith, borderWith, cv2.BORDER_CONSTANT, None, [255,255,255])
    elif(width > height):
       borderWith = (width - height) / 2
       input_image = cv2.copyMakeBorder(input_image, borderWith, borderWith, 0, 0, cv2.BORDER_CONSTANT, None,[255, 255, 255])

    input_image = cv2.resize(input_image, (dst_size, dst_size), None, 0, 0, cv2.INTER_CUBIC)
    return input_image

def merge_boundingbox(box_1, box_2):
    x_1, y_1, w_1, h_1 = box_1
    x_2, y_2, w_2, h_2 = box_2
    x_new = min(x_1, x_2)
    y_new = min(y_1, y_2)
    x_new_1 = max(x_1 + w_1, x_2 + w_2)
    y_new_1 = max(y_1 + h_1, y_2 + h_2)

    return x_new, y_new, x_new_1 - x_new, y_new_1 - y_new

class digitRecog:
    """

    """
    def __int__(self):
        self.__sess = None
        self.__model = None
        self.__input_data = None

    def init_model(self, ckpt_name):

        tf.reset_default_graph()

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.global_variables_initializer().run()}
        conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
        conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
        conv2_weights = tf.Variable(tf.truncated_normal(
            [5, 5, 32, 64], stddev=0.1,
            seed=SEED, dtype=data_type()))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
        fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                stddev=0.1,
                                seed=SEED,
                                dtype=data_type()))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
        fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                      stddev=0.1,
                                                      seed=SEED,
                                                      dtype=data_type()))
        fc2_biases = tf.Variable(tf.constant(
            0.1, shape=[NUM_LABELS], dtype=data_type()))

        def model(data, train=False):
            """The Model definition."""
            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth].
            conv = tf.nn.conv2d(data,
                                conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
            conv = tf.nn.conv2d(pool,
                                conv2_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(
                pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train:
               hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            logits = tf.matmul(hidden, fc2_weights) + fc2_biases
            return tf.nn.softmax(logits)

        self.__input_data = tf.placeholder(data_type(),shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        self.__model = model(self.__input_data)

        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        self.__sess = tf.Session(config=sess_config)
        with self.__sess.as_default():
            saver = tf.train.Saver()
            saver.restore(self.__sess, ckpt_name)

    def __process_peiyou(self, img):
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_copy = img.copy()
        save_debug_img(_DEBUG_, "binary.png", img)
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        for cnt in contours:
            (x, y, w, h) = merge_boundingbox((x,y,w,h), cv2.boundingRect(cnt))
        bouding_img = cv2.rectangle(img, (x,y), (x+w,y+h),[255, 255, 255])
        save_debug_img(_DEBUG_, "bouding.png", bouding_img)
        roi = img_copy[y:y+h, x:x+w]
        border = int(float(roi.shape[0])*0.2)
        roi = cv2.copyMakeBorder(roi, border, border, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
        roi = 255 - roi
        save_debug_img(_DEBUG_, "roi.png", roi)
        return roi

    def __preprocess(self, img):
        img = self.__process_peiyou(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not (img.shape[0] == IMAGE_SIZE and img.shape[1] == IMAGE_SIZE):
            img = get_square_image(img, IMAGE_SIZE)
            save_debug_img(_DEBUG_, "square.png",img)
            #kernel = np.ones((3, 3), np.uint8)
            #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
            #img = cv2.erode(img, kernel, iterations=1)
            #save_debug_img(_DEBUG_, "binary_morpho.png", img)
        image = np.zeros([IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)
        image[:] = (128.0 - img[:])/255.0
        image = np.reshape(image, [1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        return image

    def recog_from_file(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        return self.recog(img)

    def recog(self, img):
        img = self.__preprocess(img)
        predict = self.__sess.run(self.__model, feed_dict={self.__input_data: img})
        return predict[0]

def recog_img_files(instance, img_path, f, label=None, ext="*.png"):
    img_file_list = glob.glob(img_path + ext)
    if len(img_file_list)<1:
        print "no valid image in {}".format(img_path)
    start_time = time.time()
    right = 0
    for img_file in img_file_list:
        recog_res = recog_img(instance, img_file, f)
        if label is not None:
           right += (recog_res == int(label))
           if (recog_res != int(label)):
               shutil.copy(img_file, "./error/")
    duration_time = time.time() - start_time
    str = "avg time: {}ms".format(1000 * duration_time/len(img_file_list))
    f.write(str + "\n")
    print str
    if label is not None:
        str = "right rate {0:.3f}%".format(float(100.0 * right) / len(img_file_list))
        f.write(str)
        print str
    return right,len(img_file_list)

def recog_img(instance, img_path, f):
    start_time = time.time()
    predict = instance.recog_from_file(img_path)
    sort = np.argsort(-predict)
    recog_res = np.argmax(predict)
    confidence = predict[recog_res]
    robust = predict[sort[0]] - predict[sort[1]]
    duration_time = time.time() - start_time
    basename = os.path.basename(img_path)
    str = "{0} recog result is {1}, confidence is {2:.3f}, robust is {3:.3f}, exe time: {4}ms".format(basename, recog_res, confidence, robust, 1000 * duration_time)
    print str
    f.write(str + "\n")
    return recog_res

def recog_img_files_bath(img_file_path_base):
    total_right = 0
    total_num = 0
    for dir_path in os.listdir(img_file_path_base):
        label = dir_path
        path = os.path.join(img_file_path_base, dir_path + "/")
        result_file = path + "/digit_recog_result.txt"
        f = open(result_file, "w")
        right, num = recog_img_files(instance, path, f, label)
        total_right += right
        total_num += num
        f.close()

    str = "total image num {0}, overall right rate {1:.3f}%".format(total_num, float(100.0 * total_right) / total_num)
    print str

if __name__ == '__main__':

    ckpt_path = "model.ckpt"

    instance = digitRecog()
    instance.init_model(ckpt_name=ckpt_path)

    # img_path = "../../../official/mnist/example3.png"
    # recog_img(img_path)
    img_file_path = "../../../official/mnist/"
    img_file_path = "./error/"
    img_file_path_base = "/home/gaolining/host_share/digit/test_data/"

    result_file = "digit_recog_result.txt"
    f = open(result_file, "w")
    #recog_img_files(instance, img_file_path, f)
    f.close()

    recog_img_files_bath(img_file_path_base)

