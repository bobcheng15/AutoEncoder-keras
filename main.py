import tensorflow as tf
import cv2
import network
import numpy as np

#################################################################
#                           FLAGS                               #    
#################################################################

tf.compat.v1.flags.DEFINE_boolean("is_train", False, "training mode.")
tf.compat.v1.flags.DEFINE_boolean("restore", False, "training mode.")
# ------- Training ------- #
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "learning rate.")
tf.compat.v1.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.compat.v1.flags.DEFINE_integer("batch", 1000, "Number of images per batch.")
# ------ Save Model ------ #
tf.compat.v1.flags.DEFINE_string("ckpt_dir", "./checkpoint", "check point directory")
tf.compat.v1.flags.DEFINE_string("train_dir", "../Warehouse/atari_data/training", "Dataset directory")
tf.compat.v1.flags.DEFINE_string("test_dir", "../Warehouse/atari_data/testing", "Testing directory")
tf.compat.v1.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.compat.v1.flags.DEFINE_integer("val_num", 10000, "number of validation images.")

FLAGS = tf.compat.v1.flags.FLAGS

def get_data(is_train):
    output_img = [] # (4-d tensor) shape : size, w, h, 3
    if is_train:
        directory = FLAGS.train_dir
        st, ed = 1, 20000
    else:
        directory = FLAGS.train_dir
        st, ed = 1, FLAGS.val_num

    for i in range(st, ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        img = cv2.resize(img, (84, 84))   
        #print(img.shape)
        output_img.append(img/255.0)
    output_img = np.array(output_img)

    return output_img

def train(AE, train_data, valid_data):
    print("training...")
    for i in range (0, FLAGS.epoch):
        start, end = 0, FLAGS.batch
        while start < 20000:
            AE.model.fit(train_data[start:end], train_data[start:end], batch_size=FLAGS.batch, verbose=1)
            start += FLAGS.batch
            end += FLAGS.batch
        

if __name__ == "__main__":
    AE = network.AutoEncoder()
    AE.init()
    train_data = get_data(True)
    valid_data = get_data(False)
    if FLAGS.is_train:
        train(AE, train_data, valid_data)
        