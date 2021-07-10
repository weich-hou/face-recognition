import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, Sequential, losses
from tensorflow.keras.models import load_model

from data_process import load_dataset, resize_image

IMAGE_SIZE = 64


class Dataset:
    def __init__(self, path_name):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

        self.path_name = path_name  # 数据集加载路径
        self.user_num = len(os.listdir(path_name))

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self):
        images, labels = load_dataset(self.path_name)
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25,
                                                                                random_state=random.randint(0, 100))

        self.train_images = tf.cast(train_images, dtype=tf.float32) / 255.
        self.test_images = tf.cast(test_images, dtype=tf.float32) / 255.

        self.train_labels = tf.one_hot(tf.cast(train_labels, dtype=tf.int32), depth=self.user_num)
        self.test_labels = tf.one_hot(tf.cast(test_labels, dtype=tf.int32), depth=self.user_num)

        print(self.train_images.shape)
        print(self.train_labels.shape)

        # 输出训练集、验证集、测试集的数量
        # print('train samples:', self.train_images, self.train_labels)
        # print('test samples:', self.test_images, self.test_labels)


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, n):
        self.model = Sequential([
            # Conv-Conv-Pooling_1 32个3x3卷积核
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            layers.Dropout(rate=0.2),

            # Conv-Conv-Pooling_2 64个3x3卷积核
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            layers.Dropout(rate=0.3),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(n, activation='softmax')
        ])

        self.model.build(input_shape=[None, 64, 64, 3])
        print(self.model.summary())

    # 训练模型
    def train_net(self, datasets, batch_size, n_epoch):
        # momentum:动量参数;decay:每次更新后的学习率衰减值; nesterov:布尔值，确定是否使用nesterov动量
        self.model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
                           loss=losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.fit(datasets.train_images,
                       datasets.train_labels,
                       batch_size=batch_size,
                       epochs=n_epoch,
                       verbose=1,  # 日志显示，0:不在标准输出流输出日志信息，1:输出进度条记录，2:每个epoch输出一行记录
                       validation_split=0.2,
                       # validation_data=(datasets.valid_images, datasets.valid_labels),
                       shuffle=True)
        # validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集不参与训练，在每个epoch结束后测试模型的损失函数、精确度等指标;validation_split的划分在shuffle之前，如果数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
        # validation_data：指定的验证集,形式为tuple，此参数将覆盖validation_spilt
        """
        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))
"""

    MODEL_PATH = '../model/train_model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
        # 这种保存与加载网络的方式最为轻量级，但文件中仅保存参数张量的数值，没有其他额外的结构参数。需要使用相同的网络结构才能够恢复网络状态，一般在拥有网络源文件的情况下使用
        # self.model.save_weights(r'..\model\weights.ckpt')
        print('saved weights!')

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
        # self.model.load_weights(r'..\model\weights.ckpt')
        print('loaded weights!')

    def evaluate(self, datasets):
        score = self.model.evaluate(datasets.test_images, datasets.test_labels, verbose=1)
        print("test accuracy: %.2f%%" % (score[1] * 100))
        # print(self.model.metrics_names,score)

    # 识别人脸
    def face_predict(self, image, height, width):
        image = resize_image(image)
        image = image.reshape((1, height, width, 3))
        image = tf.cast(image, dtype=tf.float32) / 255.

        # 给出输入属于各个类别的概率
        result = self.model.predict(image)  # 返回一个n行k列的数组，[i][j]是模型预测第i个预测样本为标签j的概率，每一行概率和为1
        print('result:', result)

        if max(result[0]) >= 0.9:
            return tf.argmax(result, axis=1)
        else:
            return -1


def train(path):
    user_num = len(os.listdir(path))

    dataset = Dataset(path)
    dataset.load()

    # 训练模型
    model = Model()
    model.build_model(n=user_num)
    model.train_net(dataset, batch_size=128, n_epoch=5)
    model.save_model(file_path=r'../face_recognition/model/train_model.h5')

    # 评估模型
    model.load_model(file_path=r'../face_recognition/model/train_model.h5')
    model.evaluate(dataset)


if __name__ == '__main__':
    train(path=r'E:\Administrator\Pictures\data')
