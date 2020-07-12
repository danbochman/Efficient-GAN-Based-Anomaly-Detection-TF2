import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, ReLU
from tensorflow.keras.layers import InputLayer, Flatten, Reshape, Input, Concatenate
from tensorflow.keras.models import Model, Sequential


class BiGAN(object):
    """
    Regular Python class meant to be used as a convenient way to initialize all the moving parts and submodels necessary
    for running a Bidrectional GAN [https://arxiv.org/pdf/1605.09782.pdf]
    The model architecture is adaptive to the input_shape but (64, 64, c) or (32, 32, c) are recommended for optimal
    results.
    """

    def __init__(self, input_shape=(64, 64, 3), latent_dim=128):
        self._k_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)  # quite important
        self._latent_dim = latent_dim

        # Encoder(x) model
        self.Ex = Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer=self._k_init),
                # LeakyReLU(),  # original BiGAN paper there's no activation, in EGBAD implementation there is
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', kernel_initializer=self._k_init),
                BatchNormalization(),
                LeakyReLU(0.1),
                Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', kernel_initializer=self._k_init),
                BatchNormalization(),
                LeakyReLU(0.1),
                Flatten(),
                Dense(latent_dim, kernel_initializer=self._k_init),
            ],
            name='Ex'
        )

        self._intermediate_shapes = self.infer_intermediate_shapes()  # to preserve symmetry

        # Generator(z) model
        self.Gz = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=1024, kernel_initializer=self._k_init),  # not sure why symmetry is broken here
                BatchNormalization(),
                ReLU(),
                Dense(units=self._intermediate_shapes[1], kernel_initializer=self._k_init),
                BatchNormalization(),
                ReLU(),
                Reshape(target_shape=self._intermediate_shapes[0]),
                Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same',
                                kernel_initializer=self._k_init),
                BatchNormalization(),
                ReLU(),
                Conv2DTranspose(filters=1, kernel_size=4, strides=(2, 2), padding='same',
                                kernel_initializer=self._k_init,
                                activation='tanh'),
                #  I believe tanh is to preserve symmetry with input scaled to [-1, 1]
            ],
            name='Gz'
        )

        # Discriminator model D(x, z)
        self.Dxz = Discriminator(input_shape, latent_dim).Dxz

    def infer_intermediate_shapes(self):
        """
        Used as to infer the intermediate shapes needed for the generator model to act as an inverse to the encoder
        model
        :return tuple: (final feature map shape before flatten, after flatten)
        """
        flatten_layer = self.Ex.layers[-2]
        return flatten_layer.input_shape[1:], flatten_layer.output_shape[1:][0]


class Discriminator(object):
    """
    Class for initializing a discriminator model to be used inside a BiGAN architecture. Since the discriminator model
    for a BiGAN is rather complex by itself, wrapping its initialization in another class helps in understanding the
    general API.
    """

    def __init__(self, input_shape, latent_dim):
        self._k_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self._input_shape = input_shape
        self._latent_dim = latent_dim
        self.Dx = tf.keras.Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=self._k_init),
                LeakyReLU(0.1),
                # In EGBAD paper there's Dropout(0.5) here... not sure why... skipped
                Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=self._k_init),
                BatchNormalization(),
                LeakyReLU(0.1),
                # Also here
                Flatten(),

            ],
            name='Dx'
        )

        self.Dz = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(512, kernel_initializer=self._k_init),
                LeakyReLU(0.1),
                # Dropout here
            ],
            name='Dz'
        )

        self.Dxz = self.discriminator()

    def discriminator(self):
        """
        Part of the initialization process of the Discriminator class, strings together the Dx Dz parts to return the
        final Discriminator(x, z) model.
        :return tensorflow.keras.models.Model: Discriminator(x, z) model
        """

        x = Input(shape=self._input_shape)
        z = Input(shape=(self._latent_dim,))
        dx = self.Dx(x)
        dz = self.Dz(z)
        dxdz = Concatenate(axis=1)([dx, dz])
        intermediate_layer = Dense(1024, kernel_initializer=self._k_init)(dxdz)
        intermediate_layer = LeakyReLU(0.1)(intermediate_layer)
        # Dropout(0.5)
        logits = Dense(1, kernel_initializer=self._k_init)(intermediate_layer)
        discriminator = Model(inputs=[x, z], outputs=[logits, intermediate_layer], name='Dxz')
        return discriminator
