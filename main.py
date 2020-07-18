import tensorflow as tf

#################################################################
#                           FLAGS                               #    
#################################################################

tf.app.flags.DEFINE_boolean("is_train", False, "training mode.")
tf.app.flags.DEFINE_boolean("restore", False, "training mode.")
# ------- Training ------- #
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch", 2000, "Number of images per batch.")
# ------ Save Model ------ #
tf.app.flags.DEFINE_string("ckpt_dir", "./checkpoint", "check point directory")
tf.app.flags.DEFINE_string("train_dir", "../Warehouse/atari_data/training", "Dataset directory")
tf.app.flags.DEFINE_string("test_dir", "../Warehouse/atari_data/testing", "Testing directory")
tf.app.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.app.flags.DEFINE_integer("val_num", 16000, "number of validation images.")

FLAGS = tf.app.flags.FLAGS