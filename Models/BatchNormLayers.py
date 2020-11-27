from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model


def build_discriminator_no_batch(image_shape):
    # Discriminator attempts to classify real and generated images
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    # Leaky relu is similar to usual relu. If x < 0 then f(x) = x * alpha,
    # otherwise f(x) = x.
    model.add(LeakyReLU(alpha=0.2))

    # Dropout blocks some connections randomly. This help the model
    # to generalize better. 0.25 means that every connection has a 25%
    # chance of being blocked.
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    # Zero padding adds additional rows and columns to the image.
    # Those rows and columns are made of zeros.
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    # Flatten layer flattens the output of the previous layer to a single
    # dimension.
    model.add(Flatten())
    # Outputs a value between 0 and 1 that predicts whether image is
    # real or generated. 0 = generated, 1 = real.
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    input_image = Input(image_shape)

    # Model output given an image.
    validity = model(input_image)

    return Model(input_image, validity)
