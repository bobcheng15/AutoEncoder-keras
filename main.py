import tensorflow as tf
import cv2
import network
import numpy as np
tf.keras.losses.custom_loss = network.AutoEncoder.l2loss
#################################################################
#                           FLAGS                               #    
#################################################################

tf.compat.v1.flags.DEFINE_boolean("is_train", False, "training mode.")
tf.compat.v1.flags.DEFINE_boolean("restore", False, "training mode.")
# ------- Training ------- #
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "learning rate.")
tf.compat.v1.flags.DEFINE_integer("epoch", 50, "Number of epoch.")
tf.compat.v1.flags.DEFINE_integer("batch", 1000, "Number of images per batch.")
tf.compat.v1.flags.DEFINE_integer("start_index", 1, "The first image's index")
# ------ Save Model ------ #
tf.compat.v1.flags.DEFINE_string("ckpt_dir", "./checkpoint", "check point directory")
tf.compat.v1.flags.DEFINE_string("train_dir", "../training", "Dataset directory")
tf.compat.v1.flags.DEFINE_string("test_dir", "../Warehouse/atari_data/testing", "Testing directory")
tf.compat.v1.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.compat.v1.flags.DEFINE_integer("val_num", 10000, "number of validation images.")

FLAGS = tf.compat.v1.flags.FLAGS

def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34+160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    #print(frame.shape)
    return frame
def get_data(is_train):
    output_img = [] # (4-d tensor) shape : size, w, h, 3
    if is_train:
        directory = FLAGS.train_dir
        st, ed = FLAGS.start_index, FLAGS.start_index + 19999
    else:
        directory = FLAGS.train_dir
        st, ed = FLAGS.start_index + 20000, FLAGS.start_index + 29999

    for i in range(st, ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        img = process_frame(img)
        output_img.append(img/255)
        #print(img.shape)
    output_img = np.array(output_img, dtype=np.float32)
    print(output_img.shape)
    return output_img

def train(AE, train_data, valid_data):
    print("training...")
    for i in range (0, int(FLAGS.epoch/10)):
        #start, end = 0, FLAGS.batch * 20i
        validate(AE, valid_data)
        AE.model.save("checkPoint/AE_gray.h5")    
        #tf.keras.models.save_model(AE.decoder, "checkPoints/decoder", save_format="h5")
        AE.model.fit(train_data, train_data, epochs=10, batch_size=FLAGS.batch, verbose=1)

def validate(AE, valid_data):
    print("validating...")
    
    AE.model.evaluate(valid_data, valid_data, batch_size=1000)
            
if __name__ == "__main__":
    AE = network.AutoEncoder()
    AE.init()
    AE.model.build((None, 84, 84, 1))
    train_data = get_data(True)
    valid_data = get_data(False)
    if FLAGS.restore:
        loaded = tf.keras.models.load_model("./checkPoint/AE_gray.h5", compile=False)
        #use this line of code to only load the encoder's weight
        #AE.model.layers[0].set_weights(loaded.layers[0].get_weights())
        AE.model.set_weights(loaded.get_weights())
    if FLAGS.is_train:
        train(AE, train_data, valid_data)
    else:
        input_img = []
        img = cv2.imread(FLAGS.train_dir + "/000001.png")
        img = cv2.resize(img, (84, 84))
        print(img.shape)
        img = img / 255.0
        input_img.append(img)
        input_img = np.array(input_img)
        result = AE.model.predict(input_img)
        result = result * 255
        result.astype(int)
        print(result.shape)
        cv2.imwrite("fake.png", result[0,:,:,:])


