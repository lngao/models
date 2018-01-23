import tensorflow as tf
import numpy as np
import cv2
import time

FLAGS = None
NUM_CHANNELS = 1
IMAGE_SIZE = 28
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.

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
    height,width,channel = input_image.shape
    if(height > width):
        borderWith = (height - width)/2
        input_image = cv2.copyMakeBorder(input_image, 0, 0, borderWith, borderWith, cv2.BORDER_CONSTANT, None, [255,255,255])
    elif(width > height):
       borderWith = (width - height) / 2
       input_image = cv2.copyMakeBorder(input_image, borderWith, borderWith, 0, 0, cv2.BORDER_CONSTANT, None,[255, 255, 255])

    input_image = cv2.resize(input_image, (dst_size, dst_size), None, 0, 0, cv2.INTER_CUBIC)
    return input_image

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

        self.__sess = tf.Session()
        with self.__sess.as_default():
            saver = tf.train.Saver()
            saver.restore(self.__sess, ckpt_name)

    def __preprocess(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not (img.shape[0] == IMAGE_SIZE and img.shape[1] == IMAGE_SIZE):
            img = get_square_image(img, IMAGE_SIZE)
        img = np.reshape(img, [1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        return img

    def recog_from_file(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        return self.recog(img)

    def recog(self, img):
        img = self.__preprocess(img)
        predict = self.__sess.run(self.__model, feed_dict={self.__input_data: img})
        return predict

if __name__ == '__main__':

    img_path = "../../../official/mnist/example5.png"
    ckpt_path ="model.ckpt"

    instance = digitRecog()
    instance.init_model(ckpt_name=ckpt_path)
    start_time = time.time()
    predict = instance.recog_from_file(img_path)
    result = np.argmax(predict)
    duration_time = time.time() - start_time
    print "recog result is {}, exe time: {}ms".format(result, 1000*duration_time)