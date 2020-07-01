#问题1：当图片文件名中出现汉字时报错

import os
import cv2
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

###################官方数据载入#############################
# def load_classification_data():
#     digits = datasets.load_digits()
#     x_train = digits.data
#     y_train = digits.target
#     return train_test_split(x_train, y_train, test_size=0.25,
#                             random_state=0, stratify=y_train)
# #######################训练测试封装########################
# def test_KNeighborsClassifier(*data):
#     x_train, x_test, y_train, y_test = data
#     clf = neighbors.KNeighborsClassifier()
#     clf.fit(x_train, y_train)
#     print('training score: %f' % clf.score(x_train, y_train))
#     print('testing score: %f' % clf.score(x_test, y_test))
#############################################################

##########################图片预处理函数封装被prepro调用##########
def img_prepro(src_img):
    """
    针对一张灰度图，进行预处理,对图像进行简单的剪裁
    :param src_img: 原始图片
    :return: 预处理之后的图片
    """
    image = src_img
    # 获取长宽
    row, col = image.shape
    # 交叉遍历，顶层从底下往上找0，右边从左边往右找0
    top = row
    bottom = 0
    left = col
    right = 0
    # i为行，j为列
    for i in range(row):
        for j in range(col):
            # 找0，也就是找黑色的有字部分
            if image[i, j] == 0:
                # 找到最小的有字部分的行，也就是顶层
                if i < top:
                    top = i
                # 找到最小的有字部分的列，也就是最左边
                if j < left:
                    left = j
                # 找到最大的有字部分的行，也就是底层
                if i > bottom:
                    bottom = i
                # 找到最大的有字部分的列，也就是最右边
                if j > right:
                    right = j
    # 剪裁图像
    dst_img = image[int(top):int(bottom), int(left):int(right)]
    # 统一预处理后的图像大小
    dst_img = cv2.resize(dst_img, (8,8))
    return dst_img
################################################################
############################图片批量预处理调用######################
def prepro(dir_name , pre_dir):
    """
    根据文件夹名，循环遍历所有图像，对每一张图像进行预处理。并保存预处理后的图像。
    :param dir_name: 被遍历的文件夹名
    :param pre_dir: 预处理后的图像保存路径
    :return: 预处理过程无异常情况则返回True,否则False
    """
    # 1.获得指定文件夹下所有的文件名
    file_name_list = os.listdir(dir_name)
    # 2.针对每一个图像进行预处理操作，循环遍历文件名列表
    for file_name in file_name_list:
        # 2.1 根据文件名，读取图像（灰度图）
        img_path = dir_name + "/" + file_name
        # print(file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # 灰度图
        # 2.2 对图像进行预处理操作
        dst_img = img_prepro(img)
        # 2.3 把预处理后的图像存储
        preimg_path = pre_dir + "/" + file_name
        cv2.imwrite(preimg_path, dst_img)
    return True
#################################################################
#############单张图片特征提取被create_feature_file调用##############
def get_feature(src_img):
    """
    提取单张图片的特征值
    :param src_img:
    :return:
    """
    row, col = src_img.shape
    # 预处理后的图片8*8像素中的黑色部分就是特征值
    # 对其进行排列成1*64的行
    feature = np.array(src_img).reshape((1, row*col))
    return feature
################################################################
############################拼接特征值和目标值##################
def  create_feature_file(dir_path, data_file_name):
    """
    对dir_path下的所有图像，提取特征值，并生成数据文档
    :param dir_path: 文件夹名
    :param data_file_name: 数据文档名
    :return: 无异常，则为True
    """
    # 1.获得指定文件夹下所有的文件名
    file_name_list = os.listdir(dir_path)
    # 2.针对每一个图像进行预处理操作，循环遍历文件名列表
    X = np.zeros((1, 64))
    Y = np.zeros((1,1))
    for file_name in file_name_list:
        # 2.1 根据文件名，读取图像（灰度图）
        img_path = dir_path + "/" + file_name
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图
        # 2.2 获得当前样本的目标值
        # 根据文件的命名规则，第一个字符就是其目标值
        y = int(file_name[0])
        # 2.3 提取特征值
        feature = get_feature(img)
        # 将单个样本的目标值和特征值进行拼接，axis=0，按列拼接
        X = np.append(X, feature, axis=0)
        Y = np.append(Y, np.array(y).reshape((1,1)), axis=0)
    # 根据数据文档的特点，需要拼接X、Y。axis=1，按行拼接
    my_set = np.append(X, Y, axis=1)
    # 将数据直接保存，舍弃第一行
    np.savetxt(data_file_name, my_set[1:,:])
    return True
###########################################################################
########################提取自己的图片并按要求拆分########################
def my_load_data(file_name, test_s):
    # usecols参数意思是选取文件的列，usecols=tuple(range(64))为前64列（0-63）
    X = np.loadtxt(file_name, usecols=tuple(range(64)))
    # usecols=(64,)意思是选取第64列，也就是目标值
    Y = np.loadtxt(file_name, usecols=(64,))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_s)
    return x_train, x_test, y_train, y_test
###########################################################################
###########################训练集训练ｃｌｆ##################################
def my_train_model(x_train, y_train):
    # 调用KNN算法进行训练clf参数
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    # 用划分好的x_train, y_train进行训练
    clf.fit(x_train, y_train)
    return clf
##############################################################################
###############################测试集########################################
def my_test_model(clf, x_test, y_test):
    # 用clf自带的函数进行测试参数训练结果
    result = clf.score(x_test, y_test)
    print("测试成功率为："+ str(result))
#############################################################################
############################封装############################################
def my_app(img_path, clf):
    # 读取单张图片进行识别
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图
    # 对单张图片进行预处理
    dst_img = img_prepro(img)
    # 对单张图片提取特征
    img_feat = get_feature(dst_img)
    # 对单张图片进行预测
    result = clf.predict(img_feat)
    return result
##############################################################################
# 程序入口
if __name__ == "__main__":
    #导入处理好的特征集，若没有进行不可调用
    x_train, x_test, y_train, y_test = my_load_data('mnts_data.txt', 0)
    #获取训练好的结果ｃｌｆ
    clf = my_train_model(x_train, y_train)
    #  图像预处理，输入文件路径。可以将预处理之后的图像进行保存，以便验证预处理的结果。
    # prepro("WeMNTS", "PerMNTs")
    # #  提取特征值,并将特征值存储到数据文档中
    # create_feature_file("PerMNTs", "mnts_data.txt")
    # x_train, x_test, y_train, y_test = load_classification_data()
    # test_KNeighborsClassifier(x_train, x_test, y_train, y_test)
    # x_train1, x_test1, y_train1, y_test1 = my_load_data('mnts_data.txt', 0.3)
    # test_KNeighborsClassifier(x_train1, x_test1, y_train1, y_test1)
    # 对单张图片进行识别
    print(my_app('3_11.bmp', clf))