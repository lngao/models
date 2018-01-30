import tensorflow as tf
import numpy as np
import cv2
import time
import glob
import os
import shutil

# some const params
NUM_CHANNELS = 1
IMAGE_SIZE = 28
NUM_LABELS = 10
_DEBUG_ = True

#some pulic funcitons
def save_debug_img(is_debug, img_path, img):
    """

    :param is_debug:
    :param img_path:
    :param img:
    :return:
    """
    if is_debug:
        cv2.imwrite(img_path, img)

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
    """

    :param box_1:
    :param box_2:
    :return:
    """
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
        """

        :return:
        """
        self.__sess = None
        self.__model = None
        self.__input_data = None
        self.__inited = False

    def init_model(self, graph_name):
        """
        model init
        :param graph_name: the graph def file path
        :return:
        """
        self.__inited = False
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_name, "rb") as f:
            graph_def.ParseFromString(f.read())
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def)
            self.__input_data = tf.get_default_graph().get_tensor_by_name("import/Placeholder:0")
            self.__model = tf.get_default_graph().get_tensor_by_name("import/Softmax:0")
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
            self.__sess = tf.Session(config=sess_config, graph=graph)

        self.__inited = True
        return self.__inited

    def __process_peiyou(self, img):
        """

        :param img:
        :return:
        """
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mean_value = np.mean(img)
        if mean_value > 128:
            img = 255 - img
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
        """

        :param img:
        :return:
        """
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
        """

        :param img_path:
        :return:
        """
        if not self.__inited:
            print "model not inited"
            return None
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return self.recog(img)

    def recog(self, img):
        """

        :param img:
        :return:
        """
        if img is None:
            return ""
        img = self.__preprocess(img)
        predict = self.__sess.run(self.__model, feed_dict={self.__input_data: img})
        predict = predict[0]
        recog_res = np.argmax(predict)
        sort = np.argsort(-predict)
        confidence = predict[recog_res]
        robust = predict[sort[0]] - predict[sort[1]]
        return recog_res,confidence,robust

# functions show how to use digitRecog
def recog_img_file(instance, img_path, f):
    """

    :param instance:
    :param img_path:
    :param f:
    :return:
    """
    start_time = time.time()
    recog_res, confidence, robust = instance.recog_from_file(img_path)
    duration_time = time.time() - start_time
    basename = os.path.basename(img_path)
    str = "{0} recog result is {1}, confidence is {2:.3f}, robust is {3:.3f}, exe time: {4}ms".format(basename,
                                                                                                      recog_res,
                                                                                                      confidence,
                                                                                                      robust,
                                                                                                      1000 * duration_time)
    print str
    f.write(str + "\n")
    return recog_res

def recog_img_files(instance, img_path, f, label=None, ext="*.png"):
    """

    :param instance:
    :param img_path:
    :param f:
    :param label:
    :param ext:
    :return:
    """
    img_file_list = glob.glob(img_path + ext)
    if len(img_file_list)<1:
        print "no valid image in {}".format(img_path)
    start_time = time.time()
    right = 0
    for img_file in img_file_list:
        recog_res = recog_img_file(instance, img_file, f)
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

def recog_multi_img_files(img_file_path_base):
    """

    :param img_file_path_base:
    :return:
    """
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

    # the graph def binary files define the model
    graph_name = "frozen_digit.pb"

    # inital the model
    instance = digitRecog()
    instance.init_model(graph_name=graph_name)

    # do recognize
    result_file = "digit_recog_result.txt"
    f = open(result_file, "w")
    recognize_tpye = 2
    if recognize_tpye == 0:
        # recognize single image
        img_path = "../../../official/mnist/example3.png"
        recog_img_file(instance, img_path, f)
    elif recognize_tpye ==1:
        # recognize dir images
        img_file_path = "../../../official/mnist/"
        recog_img_files(instance, img_file_path, f, None)
    else:
        # recognize multi dirs
        img_file_path_base = "/home/gaolining/host_share/digit/test_data/"
        recog_multi_img_files(img_file_path_base)
    f.close()
