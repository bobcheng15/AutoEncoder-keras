import tensorflow as tf

#################################################################
#                           PARAMETERS                          #    
#################################################################
CONV1_FILTERS = 32
CONV2_FILTERS = 64
CONV3_FILTERS = 64

#################################################################
#                           ENCODER                             #    
#################################################################

class AutoEncoder_en():
    def __init__(self):
        #encoder
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(CONV1_FILTERS, 8, [4, 4], activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(CONV1_FILTERS, 8, [4, 4], activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(CONV1_FILTERS, 8, [4, 4], activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(), 
        ])

#################################################################
#                           DECODER                             #    
#################################################################

class AutoEncoder_de():
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(CONV1_FILTERS, 8, [4, 4], activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(CONV1_FILTERS, 8, [4, 4], activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(CONV1_FILTERS, 8, [4, 4], activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(), 
        ])

class AutoEncoder():
    def __init__(self):
        self.encoder = AutoEncoder_en()
        self.decoder = AutoEncoder_de()
        self.model = tf.keras.models.Sequential([
            self.encoder.model,
            self.decoder.model,
        ])

