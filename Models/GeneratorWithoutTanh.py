from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, BatchNormalization, Activation


def build_generator_without_tanh(random_noise_dimension, channels):
    """
    Generator attempts to fool discriminator by generating new images.
    This is modified version fo generator form Dofo Week 6 with removed Tanh

    Parameters
    ----------
    random_noise_dimension: dimesion of moise matrix
    channels: number of channels

    Returns
    -------
    Generator
    """
    model = Sequential()

    model.add(Dense(256 * 4 * 4, activation="relu",
                    input_dim=random_noise_dimension))
    model.add(Reshape((4, 4, 256)))

    # Four layers of upsampling, convolution, batch normalization
    # and activation.
    # 1. Upsampling: Input data is repeated. Default is (2,2).
    #    In that case a 4x4x256 array becomes an 8x8x256 array.
    # 2. Convolution: If you are not familiar, you should watch
    #    this video: https://www.youtube.com/watch?v=FTr3n7uBIuE
    # 3. Normalization normalizes outputs from convolution.
    # 4. Relu activation:  f(x) = max(0,x). If x < 0, then f(x) = 0.

    # example2  Add another conv block - after removing the 1st upsampling Layer
    # copy of next block after deleting upsampling layer
    # may adjust 256
    # model.add(Conv2D(256,kernel_size=3,padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Last convolutional layer outputs as many featuremaps as channels
    # in the final image.
    model.add(Conv2D(channels, kernel_size=3, padding="same"))

    # show the summary of the model architecture
    model.summary()

    # Placeholder for the random noise input
    model_input = Input(shape=(random_noise_dimension,))
    # Model output
    generated_image = model(model_input)

    # Change the model type from Sequential to Model (functional API)
    # More at: https://keras.io/models/model/.
    return Model(model_input, generated_image)
