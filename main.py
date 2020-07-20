import tensorflow as tf
import cv2
import network

#################################################################
#                           FLAGS                               #    
#################################################################

tf.compat.v1.flags.DEFINE_boolean("is_train", False, "training mode.")
tf.compat.v1.flags.DEFINE_boolean("restore", False, "training mode.")
# ------- Training ------- #
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "learning rate.")
tf.compat.v1.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.compat.v1.flags.DEFINE_integer("batch", 2000, "Number of images per batch.")
# ------ Save Model ------ #
tf.compat.v1.flags.DEFINE_string("ckpt_dir", "./checkpoint", "check point directory")
tf.compat.v1.flags.DEFINE_string("train_dir", "../Warehouse/atari_data/training", "Dataset directory")
tf.compat.v1.flags.DEFINE_string("test_dir", "../Warehouse/atari_data/testing", "Testing directory")
tf.compat.v1.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.compat.v1.flags.DEFINE_integer("val_num", 16000, "number of validation images.")

FLAGS = tf.compat.v1.flags.FLAGS

def get_data(is_train=FLAGS.is_train):
    output_img = [] # (4-d tensor) shape : size, w, h, 3
    if is_train:
        directory = FLAGS.train_dir
        st, ed = 1, 50000
    else:
        directory = FLAGS.test_dir
        st, ed = 0, FALGS.val_num

    for i in range(st, ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        cv2.resize(img, (84, 84))
        output_img.append(img/255.0)

    return output_img

def train(model, train_data, valid_data):
    print("training...")
    batch_loss = 0
    st, ed, times = 0, 0, 0
    max_len = len(data)
    r = 0
    for i in range (0, FLAG.epoch):
        model.fit(data, data, FLAGS.batch, validation_data=(data, data), validation_batch_size=FLAGS.batch, verbose=1, return_dict=True)


if __name__ == "__main__":
    model = network.AutoEncoder()
    train_data = get_data()
    valid_data = get_data(False)
    if FLAGS.is_train:
        train(model, train_data, valid_data)
        