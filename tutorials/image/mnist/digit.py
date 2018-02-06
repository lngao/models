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
        cv2.imwrite("./tmp/" + img_path, img)

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

def overlap(box_1, box_2):
    h_thld_l = 0.8
    h_thld_s = 0.4
    v_thld = 0.4

    x_1, y_1, w_1, h_1 = box_1
    x_2, y_2, w_2, h_2 = box_2

    if (x_1 > x_2) and ((x_1 + w_1) < (x_2 + w_2)):
        return True

    if (x_2 > x_1) and ((x_1 + w_1) > (x_2 + w_2)):
        return True

    dis_x = abs((x_1 + w_1/2.0) - (x_2 + w_2/2.0))
    horizental_overlap = max(0, 1 - dis_x/(w_1/2.0 + w_2/2.0))

    dis_y = abs((y_1 + h_1 / 2.0) - (y_2 + h_2 / 2.0))
    vertical_overlap = max(0, 1 - dis_y / (w_1 / 2.0 + w_2 / 2.0))

    # if horizental overlap is very large or (horizental is large and vertial overlap is small)
    return horizental_overlap > h_thld_l or (horizental_overlap > h_thld_s and vertical_overlap < v_thld)

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
        mean_value = np.mean(img)#cv::mean
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

    def __get_roi_img(self, contour_list, flag_list, img):

        roi_list = []
        max_height = 0
        for contour in contour_list:
            if( contour[0][3] > max_height):
                max_height = contour[0][3]

        for i, contour in enumerate(contour_list):
            if not flag_list[i]:
                continue
            (x, y, w, h) = contour[0]

            # check the box validation
            h_ratio = float(h) / max_height
            if h_ratio < 0.3 :
                print "invalid h_{}".format(h_ratio)
                continue

            # get the roi image by box and contour mask
            bouding_img = img.copy()
            bouding_mask = img.copy()
            bouding_mask[:] = 0
            cv2.drawContours(bouding_mask, [contour[1]], 0, (255, 255, 255), -1)
            bouding_img = cv2.bitwise_and(bouding_img, bouding_mask)

            # if the width/height ratio is too large ,may be two number linked, so segement to two by center
            ratio = float(w) / h
            if ratio > 1.2:
                print ratio
                roi = bouding_img[y:y + h, x:x + w/2]
                border = int(float(h) * 0.2)
                roi = cv2.copyMakeBorder(roi, border, border, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
                roi = 255 - roi
                save_debug_img(_DEBUG_, "roi_0_{}.png".format(i), roi)
                roi_list.append(roi)

                roi = bouding_img[y:y + h, x + w/2:x + w]
                border = int(float(h) * 0.2)
                roi = cv2.copyMakeBorder(roi, border, border, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
                roi = 255 - roi
                save_debug_img(_DEBUG_, "roi_1_{}.png".format(i), roi)
                roi_list.append(roi)

            else:
                roi = bouding_img[y:y + h, x:x + w]
                #border = int(float(roi.shape[0]) * 0.2)
                border = int(float(h) * 0.2)
                roi = cv2.copyMakeBorder(roi, border, border, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
                roi = 255 - roi
                save_debug_img(_DEBUG_, "roi_{}.png".format(i), roi)
                roi_list.append(roi)
        return roi_list

    def __segment(self, img):
        """

        :param img:
        :return:
        """
        # binary image
        # thld_list = [0,10,20,50,100,200,220]
        # for thld in thld_list:
        #     ret, tmp = cv2.threshold(img.copy(), thld, 255, cv2.THRESH_BINARY)
        #     save_debug_img(_DEBUG_, "binary_{}.png".format(thld), tmp)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#

        # make sure the backgroud is black, if not reverse the image
        mean_value = np.mean(img)  # cv::mean
        if mean_value > 128:
            img = 255 - img
        save_debug_img(_DEBUG_, "binary.png", img)

        # find contours
        # method_list = [cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_TC89_KCOS, cv2.CHAIN_APPROX_TC89_L1]
        # for i,method in enumerate(method_list):
        #     _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, method)
        #     bouding_img = img.copy()
        #     for cnt in contours:
        #         (x, y, w, h) = cv2.boundingRect(cnt)
        #         bouding_img = cv2.rectangle(bouding_img, (x, y), (x + w, y + h), [255, 255, 255])
        #     save_debug_img(_DEBUG_, "bouding_{}.png".format(i), bouding_img)
        _, contonurs, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contonurs) < 1):
            return None

        contour_list=[]
        for cnt in contonurs:
            (x, y, w, h) = cv2.boundingRect(cnt)
            contour_list.append([(x, y, w, h), cnt])

        if _DEBUG_:
            bouding_img = img.copy()
            bouding_img[:] = 0
            for contour in contour_list:
                (x, y, w, h) = contour[0]
                cnt = contour[1]
                bouding_img = cv2.rectangle(bouding_img, (x, y), (x + w, y + h), [255, 255, 255])
                cv2.drawContours(bouding_img, [cnt], 0, (255, 255, 255), -1)
            save_debug_img(_DEBUG_, "bouding.png", bouding_img)

        # sort box by x
        contour_list.sort(key=lambda x:x[0][0])

        # merge overlap box
        flag_list = []
        for i in range(len(contour_list)):
            flag_list.append(True)
        for i in range(len(contour_list)-1):
            if not flag_list[i]:
                continue
            for j in range(i+1, len(contour_list)):
                if not flag_list[j]:
                    continue
                if overlap(contour_list[i][0], contour_list[j][0]):
                    merge_box = merge_boundingbox(contour_list[i][0], contour_list[j][0])
                    contour_list[i][0] = merge_box
                    flag_list[j]=False
        # get roi img list
        roi_img_list = self.__get_roi_img(contour_list, flag_list, img)
        return  roi_img_list

    def __preprocess(self, img):
        """

        :param img:
        :return:
        """
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = self.__segment(img)
        if not img_list:
            return []

        image_list = []
        for i,img in enumerate(img_list):
            if not (img.shape[0] == IMAGE_SIZE and img.shape[1] == IMAGE_SIZE):
                img = get_square_image(img, IMAGE_SIZE)
                save_debug_img(_DEBUG_, "square_{}.png".format(i),img)
            image = np.zeros([IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)
            image[:] = (128.0 - img[:])/255.0#cv::convert
            image = np.reshape(image, [1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            image_list.append(image)
        return image_list

    def recog_from_file(self, img_path):
        """

        :param img_path:
        :return:
        """
        if not self.__inited:
            print "model not inited"
            return None
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if _DEBUG_:
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp")
            os.mkdir("./tmp")
        return self.recog(img)

    def recog(self, img):
        """

        :param img:
        :return:
        """
        if img is None:
            return ""
        img_list = self.__preprocess(img)
        if len(img_list) < 1:
            return "0", 0, 0
        recog_res = ""
        confidence = 0.0
        robust = 0.0
        for img in img_list:
            predict = self.__sess.run(self.__model, feed_dict={self.__input_data: img})
            predict = predict[0]
            recog_res += str(np.argmax(predict))
            sort = np.argsort(-predict)
            confidence += predict[sort[0]]
            robust += predict.tolist()[sort[0]] - predict[sort[1]]
        confidence/=len(img_list)
        robust/=len(img_list)
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
           right += (recog_res == label)
           if (recog_res != label):
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
    recognize_tpye = 1
    if recognize_tpye == 0:
        # recognize single image
        img_path = "../../../official/mnist/example3.png"
        recog_img_file(instance, img_path, f)
    elif recognize_tpye ==1:
        # recognize dir images
        img_file_path = "/home/gaolining/host_share/digit/samples_m/test/"
        recog_img_files(instance, img_file_path, f, None)
    else:
        # recognize multi dirs
        img_file_path_base = "/home/gaolining/host_share/digit/samples_m/"
        recog_multi_img_files(img_file_path_base)
    f.close()
