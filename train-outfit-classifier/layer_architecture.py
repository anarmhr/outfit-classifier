from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, MaxPooling2D, \
    GlobalAveragePooling2D

# Index: Architecture name
# 0: Default
# 1: Alexa Net

layer_array = [
    [
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)),
        MaxPool2D(2, 2),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPool2D(2, 2),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPool2D(),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(46, activation="softmax"),
    ],

    [
        Conv2D(96, kernel_size=(11, 11), strides=4,
               padding='valid', activation='relu',
               input_shape=(256, 256, 3),
               kernel_initializer='he_normal'),

        MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                     padding='valid', data_format=None),

        Conv2D(256, kernel_size=(5, 5), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),

        MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                     padding='valid', data_format=None),

        Conv2D(384, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),

        Conv2D(256, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),

        MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                     padding='valid', data_format=None),

        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(46, activation='softmax')
    ]
]
