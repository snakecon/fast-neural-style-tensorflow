# coding=utf-8
import os

import tensorflow as tf
import time
import cv2


model_dir = '.'


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def predict():
    output_graph_path = os.path.join(model_dir, "wave.pb")
    graph = load_graph(output_graph_path)

    for op in graph.get_operations():
        print(op.name)

    input_image = graph.get_tensor_by_name('prefix/input_image:0')
    # height = graph.get_tensor_by_name('prefix/height:0')
    # width = graph.get_tensor_by_name('prefix/width:0')
    output_image = graph.get_tensor_by_name('prefix/output_image:0')


    orig_img = cv2.imread('img/test1.jpg')
    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    with tf.Session(graph=graph) as sess:
        start_time = time.time()
        y_out = sess.run(output_image, feed_dict={input_image: rgb_img})
        end_time = time.time()
        print('Elapsed time: %fs' % (end_time - start_time))

        # with open('res.jpg', 'wb') as img:
        #     img.write(y_out)
        print(y_out.shape)

        cv2.imwrite('result.jpg', rgb2bgr(y_out))



def rgb2bgr(img):
    r, g, b = cv2.split(img)
    return cv2.merge([b, g, r])


def main(args):
    predict()


if __name__ == "__main__":
    tf.app.run()
