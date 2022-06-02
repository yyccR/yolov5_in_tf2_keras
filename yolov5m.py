import sys

sys.path.append("../yolov5_in_tf2_keras")

import tensorflow as tf
from layers import Conv, C3, SPPF, Concat

class Yolov5m:
    def __init__(self, image_shape, batch_size, num_class, anchors_per_location):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.num_class = num_class
        self.anchors_per_location = anchors_per_location

    def build_graph(self):
        """
        :param inputs:
        :return: p7: [batch, h/8, w/8, anchors, num_class+5]
                 p8: [batch, h/16, w/16, anchors, num_class+5]
                 p9: [batch, h/32, w/32, anchors, num_class+5]
        """
        inputs = tf.keras.Input(shape=self.image_shape, batch_size=self.batch_size)
        # backbone
        x = Conv(out_channels=48, kernel_size=6, stride=2, padding='same')(inputs)
        # 1/4
        x = Conv(out_channels=96, kernel_size=3, stride=2, padding='same')(x)
        x = C3(out_channels=96, num_bottles=2)(x)
        # 1/8
        p3 = x = Conv(out_channels=192, kernel_size=3, stride=2, padding='same')(x)
        x = C3(out_channels=192, num_bottles=4)(x)
        # 1/16
        p4 = x = Conv(out_channels=384, kernel_size=3, stride=2, padding='same')(x)
        x = C3(out_channels=384, num_bottles=6)(x)
        # 1/32
        x = Conv(out_channels=768, kernel_size=3, stride=2, padding='same')(x)
        x = C3(out_channels=768, num_bottles=2)(x)
        x = SPPF(in_channels=768, out_channels=768, kernel_size=5)(x)

        # head
        p5 = x = Conv(out_channels=384, kernel_size=1, stride=1)(x)
        # 1/16
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = Concat(dimension=3)([x, p4])
        x = C3(out_channels=384, num_bottles=2, shortcut=False)(x)
        p6 = x = Conv(out_channels=192, kernel_size=1, stride=1, padding='same')(x)
        # 1/8
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = Concat(dimension=3)([x, p3])
        p7 = x = C3(out_channels=192, num_bottles=2, shortcut=False)(x)
        # 1/16
        x = Conv(out_channels=192, kernel_size=3, stride=2, padding='same')(x)
        x = Concat(dimension=3)([x, p6])
        p8 = x = C3(out_channels=384, num_bottles=2, shortcut=False)(x)
        # 1/32
        x = Conv(out_channels=384, kernel_size=3, stride=2, padding='same')(x)
        x = Concat(dimension=3)([x, p5])
        p9 = C3(out_channels=768, num_bottles=2, shortcut=False)(x)

        # output tensor [batch, grid, grid, anchors, 5 + num_classes]
        p7 = tf.keras.layers.Conv2D((self.num_class + 5) * self.anchors_per_location, kernel_size=1)(p7)
        # p7_shape = tf.shape(p7)
        # p7 = tf.reshape(p7, [self.batch_size, p7_shape[1], p7_shape[2], self.anchors_per_location, self.num_class + 5])
        p7 = tf.keras.layers.Reshape([self.image_shape[0]//8, self.image_shape[1]//8, self.anchors_per_location, self.num_class + 5])(p7)

        # [batch, grid, grid, anchors, 5 + num_classes]
        p8 = tf.keras.layers.Conv2D((self.num_class + 5) * self.anchors_per_location, kernel_size=1)(p8)
        # p8_shape = tf.shape(p8)
        # p8 = tf.reshape(p8, [self.batch_size, p8_shape[1], p8_shape[2], self.anchors_per_location, self.num_class + 5])
        p8 = tf.keras.layers.Reshape([self.image_shape[0]//16, self.image_shape[1]//16, self.anchors_per_location, self.num_class + 5])(p8)

        # [batch, grid, grid, anchors, 5 + num_classes]
        p9 = tf.keras.layers.Conv2D((self.num_class + 5) * self.anchors_per_location, kernel_size=1)(p9)
        # p9_shape = tf.shape(p9)
        # p9 = tf.reshape(p9, [self.batch_size, p9_shape[1], p9_shape[2], self.anchors_per_location, self.num_class + 5])
        p9 = tf.keras.layers.Reshape([self.image_shape[0]//32, self.image_shape[1]//32, self.anchors_per_location, self.num_class + 5])(p9)

        model = tf.keras.models.Model(inputs=inputs, outputs=[p7, p8, p9])
        return model


def gen_data():
    while True:
        image = tf.random.normal([2, 512, 512, 3])
        p7 = tf.random.normal([2, 512 // 8, 512 // 8, 256])
        p8 = tf.random.normal([2, 512 // 16, 512 // 16, 512])
        p9 = tf.random.normal([2, 512 // 32, 512 // 32, 1024])
        yield image, [p7, p8, p9]


if __name__ == "__main__":
    # for i,o in ds_series:
    #     print(tf.shape(i))
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # from tensorflow.python.ops import summary_ops_v2
    # from tensorflow.python.keras.backend import get_graph
    yolo5m = Yolov5m(image_shape=(416, 416, 3),
                     batch_size=2,
                     num_class=30,
                     anchors_per_location=3)
    model = yolo5m.build_graph()
    model.summary(line_length=200)
    # model.compile(
    #     optimizer='adam',
    #     loss='mse',
    #     metrics=['accuracy'])
    # tb_writer = tf.summary.create_file_writer('./logs')
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    # model.fit_generator(iter(gen_data()), steps_per_epoch=10, epochs=1, callbacks=[tensorboard_callback])
    # tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    # z = my_func(x, y)
    # with tb_writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir='./logs')
