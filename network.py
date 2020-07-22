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
            tf.keras.layers.Conv2D(CONV1_FILTERS, 8, [4, 4], padding="valid", activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(CONV2_FILTERS, 4, [2, 2], padding="valid", activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(CONV3_FILTERS, 3, [1, 1], padding="valid", activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(), 
        ])

#################################################################
#                           DECODER                             #    
#################################################################

class AutoEncoder_de():
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(CONV2_FILTERS, 3, [1, 1], padding="valid", activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(CONV1_FILTERS, 4, [2, 2], padding="valid", activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(1, 8, [4, 4], padding="valid", activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(), 
        ])

class AutoEncoder():

    def __init__(self):
        self.lr = 0.001
        self.encoder = AutoEncoder_en()
        self.decoder = AutoEncoder_de()
        self.model = tf.keras.models.Sequential([
            self.encoder.model,
            self.decoder.model,
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.95)
    
    def l2loss(self, y_true, y_pred):
        tf.keras.backend.clip(y_pred, 0.0, 1.0)
        return (y_true - y_pred) ** 2
    def init(self):
        self.model.compile(optimizer=self.optimizer, loss=self.l2loss, metrics=['accuracy'])

#AutoEncoder()
