"""Arquitecturas generativas para Fashion-MNIST (28x28).

El módulo expone dos familias de modelos orientadas a objetos:

* :class:`Autoencoder` -- autoencoder tradicional (determinista).
* :class:`VariationalAutoencoder` -- VAE con *Reparameterization Trick* y
  pérdida combinada (reconstrucción + divergencia KL).

Ambos comparten los mismos bloques de codificación/decodificación, que pueden
ser puramente densos (``conv=False``) o convolucionales (``conv=True``).

Espacio latente
---------------
* ``Autoencoder``: espacio latente **determinista** ``z in R^latent_dim``.
  ``encode`` produce un único vector; se usa ``decode`` para reconstruir.
* ``VariationalAutoencoder``: espacio latente **probabilístico**; el encoder
  ``q(z|x)`` parametriza una gaussiana ``N(z_mean, exp(z_log_var))`` de
  dimensión ``latent_dim`` y ``reparameterize`` muestrea ``z`` de forma
  diferenciable. La prior es ``N(0, I)``, lo que hace que la KL regularice la
  media hacia 0 y la varianza hacia 1.
"""

from __future__ import annotations

import tensorflow as tf

IMAGE_SHAPE: tuple[int, int, int] = (28, 28, 1)
IMAGE_SIZE: int = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
OUTPUT_ACTIVATION = "sigmoid"


def _encoder_features(inputs, conv: bool, base_filters: int, hidden_units: int):
    """Extrae características de una imagen 28x28x1 (bloque compartido)."""
    if conv:
        values = tf.keras.layers.Conv2D(
            base_filters, 3, strides=2, padding="same", activation="relu", name="enc_conv_1"
        )(inputs)
        values = tf.keras.layers.Conv2D(
            base_filters * 2, 3, strides=2, padding="same", activation="relu", name="enc_conv_2"
        )(values)
        values = tf.keras.layers.Flatten(name="enc_flatten")(values)
    else:
        values = tf.keras.layers.Flatten(name="enc_flatten")(inputs)
    return tf.keras.layers.Dense(hidden_units, activation="relu", name="enc_dense")(values)


def build_encoder(
    latent_dim: int = 16,
    conv: bool = False,
    base_filters: int = 32,
    hidden_units: int = 256,
    name: str = "encoder",
):
    """Encoder determinista ``x -> z`` del autoencoder tradicional."""
    inputs = tf.keras.Input(IMAGE_SHAPE, name="image")
    features = _encoder_features(inputs, conv, base_filters, hidden_units)
    latent = tf.keras.layers.Dense(latent_dim, name="latent")(features)
    return tf.keras.Model(inputs, latent, name=name)


def build_variational_encoder(
    latent_dim: int = 2,
    conv: bool = False,
    base_filters: int = 32,
    hidden_units: int = 256,
    name: str = "encoder",
):
    """Encoder variacional ``x -> (z_mean, z_log_var)`` que define ``q(z|x)``."""
    inputs = tf.keras.Input(IMAGE_SHAPE, name="image")
    features = _encoder_features(inputs, conv, base_filters, hidden_units)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(features)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(features)
    return tf.keras.Model(inputs, [z_mean, z_log_var], name=name)


def build_decoder(
    latent_dim: int = 16,
    conv: bool = False,
    base_filters: int = 32,
    hidden_units: int = 256,
    name: str = "decoder",
):
    """Decoder ``z -> x_hat`` compartido por el AE y el VAE."""
    inputs = tf.keras.Input((latent_dim,), name="latent")
    if conv:
        values = tf.keras.layers.Dense(7 * 7 * base_filters, activation="relu", name="dec_dense")(inputs)
        values = tf.keras.layers.Reshape((7, 7, base_filters), name="dec_reshape")(values)
        values = tf.keras.layers.Conv2DTranspose(
            base_filters * 2, 3, strides=2, padding="same", activation="relu", name="dec_deconv_1"
        )(values)
        values = tf.keras.layers.Conv2DTranspose(
            base_filters, 3, strides=2, padding="same", activation="relu", name="dec_deconv_2"
        )(values)
        outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation=OUTPUT_ACTIVATION, name="dec_output")(values)
    else:
        values = tf.keras.layers.Dense(hidden_units, activation="relu", name="dec_dense")(inputs)
        values = tf.keras.layers.Dense(IMAGE_SIZE, activation=OUTPUT_ACTIVATION, name="dec_output")(values)
        outputs = tf.keras.layers.Reshape(IMAGE_SHAPE, name="dec_reshape")(values)
    return tf.keras.Model(inputs, outputs, name=name)


@tf.keras.utils.register_keras_serializable(package="inf8239_u03_gen")
class Autoencoder(tf.keras.Model):
    """Autoencoder tradicional con espacio latente determinista ``R^latent_dim``.

    La pérdida de entrenamiento es únicamente de reconstrucción
    (entropía cruzada binaria); no modela densidad, por lo que no permite
    muestrear de forma natural. Para generar se decodifican puntos arbitrarios
    (o ``sample``) del espacio latente.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        conv: bool = False,
        base_filters: int = 32,
        hidden_units: int = 256,
        name: str = "autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = int(latent_dim)
        self.conv = bool(conv)
        self.base_filters = int(base_filters)
        self.hidden_units = int(hidden_units)
        self.encoder = build_encoder(self.latent_dim, self.conv, self.base_filters, self.hidden_units)
        self.decoder = build_decoder(self.latent_dim, self.conv, self.base_filters, self.hidden_units)

    # -- Espacio latente -------------------------------------------------
    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Forma del vector latente determinista (una dimensión por variable)."""
        return (self.latent_dim,)

    # -- API de alto nivel ----------------------------------------------
    def encode(self, images, training: bool = False):
        """Proyecta imágenes al espacio latente determinista."""
        return self.encoder(images, training=training)

    def decode(self, latent, training: bool = False):
        """Reconstruye imágenes a partir de códigos latentes."""
        return self.decoder(latent, training=training)

    def sample(self, n: int, seed: int | None = None):
        """Decodifica ``n`` puntos aleatorios ``N(0, I)`` del espacio latente."""
        latent = tf.random.normal(shape=(n, self.latent_dim), seed=seed)
        return self.decode(latent)

    def call(self, inputs, training=None):
        return self.decode(self.encode(inputs, training=training), training=training)

    # -- Pérdida y serialización ----------------------------------------
    @staticmethod
    def reconstruction_loss(images, reconstruction):
        """BCE sumada sobre los píxeles y promediada sobre el lote.

        En Keras 3 ``binary_crossentropy`` ya reduce el eje del canal
        (``axis=-1``), por lo que se suman las dimensiones espaciales.
        """
        bce = tf.keras.losses.binary_crossentropy(images, reconstruction)
        return tf.reduce_mean(tf.reduce_sum(bce, axis=(1, 2)))

    @property
    def reconstruction_model(self):
        """Modelo funcional ``imagen -> reconstrucción`` (serializable)."""
        inputs = tf.keras.Input(IMAGE_SHAPE, name="image")
        return tf.keras.Model(inputs, self.decoder(self.encoder(inputs)), name=self.name)

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "conv": self.conv,
            "base_filters": self.base_filters,
            "hidden_units": self.hidden_units,
            "name": self.name,
        }


@tf.keras.utils.register_keras_serializable(package="inf8239_u03_gen")
class VariationalAutoencoder(tf.keras.Model):
    """VAE con *Reparameterization Trick* y pérdida combinada.

    Definiciones clave:

    * Encoder ``q(z|x) = N(z_mean, diag(exp(z_log_var)))``.
    * Prior ``p(z) = N(0, I)``.
    * Reparametrización ``z = z_mean + exp(0.5 * z_log_var) * eps``, con
      ``eps ~ N(0, I)``. Así el muestreo es diferenciable y el gradiente
      fluye hacia ``z_mean`` y ``z_log_var``.
    * Pérdida ``L = L_rec + kl_weight * KL(q(z|x) || p(z))``, donde
      ``KL = -0.5 * sum(1 + z_log_var - z_mean^2 - exp(z_log_var))``.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        encoder=None,
        decoder=None,
        conv: bool = False,
        base_filters: int = 32,
        hidden_units: int = 256,
        kl_weight: float = 1.0,
        name: str = "vae",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = int(latent_dim)
        self.conv = bool(conv)
        self.base_filters = int(base_filters)
        self.hidden_units = int(hidden_units)
        self.kl_weight = float(kl_weight)
        self.encoder = (
            encoder
            if encoder is not None
            else build_variational_encoder(self.latent_dim, self.conv, self.base_filters, self.hidden_units)
        )
        self.decoder = (
            decoder
            if decoder is not None
            else build_decoder(self.latent_dim, self.conv, self.base_filters, self.hidden_units)
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    # -- Espacio latente -------------------------------------------------
    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Dimensión del vector latente gaussiano (media y varianza)."""
        return (self.latent_dim,)

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    # -- Reparameterization Trick ---------------------------------------
    def encode(self, images, training: bool = False):
        """Devuelve ``(z_mean, z_log_var)`` que parametrizan ``q(z|x)``."""
        return self.encoder(images, training=training)

    def reparameterize(self, z_mean, z_log_var):
        """Truco de reparametrización: ``z = mu + sigma * eps``, ``eps ~ N(0, I)``."""
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, latent, training: bool = False):
        """Decodifica códigos latentes a imágenes."""
        return self.decoder(latent, training=training)

    def sample(self, n: int, seed: int | None = None):
        """Genera ``n`` imágenes muestreando del prior ``N(0, I)``."""
        latent = tf.random.normal(shape=(n, self.latent_dim), seed=seed)
        return self.decode(latent)

    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encode(inputs, training=training)
        return self.decode(self.reparameterize(z_mean, z_log_var), training=training)

    # -- Pérdida combinada ----------------------------------------------
    @staticmethod
    def reconstruction_loss(images, reconstruction):
        """Término de reconstrucción: BCE sumada sobre píxeles y promediada.

        En Keras 3 ``binary_crossentropy`` ya reduce el eje del canal
        (``axis=-1``), por lo que se suman las dimensiones espaciales.
        """
        bce = tf.keras.losses.binary_crossentropy(images, reconstruction)
        return tf.reduce_mean(tf.reduce_sum(bce, axis=(1, 2)))

    @staticmethod
    def kl_divergence(z_mean, z_log_var):
        """KL(N(mu, exp(log_var)) || N(0, I)) promediada sobre el lote."""
        kl = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        return -0.5 * tf.reduce_mean(tf.reduce_sum(kl, axis=1))

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(images, training=True)
            latent = self.reparameterize(z_mean, z_log_var)
            reconstruction = self.decode(latent, training=True)
            reconstruction_loss = self.reconstruction_loss(images, reconstruction)
            kl_loss = self.kl_divergence(z_mean, z_log_var)
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "conv": self.conv,
            "base_filters": self.base_filters,
            "hidden_units": self.hidden_units,
            "kl_weight": self.kl_weight,
            "name": self.name,
        }


# -- Constructores de compatibilidad (API funcional original) -------------
def build_autoencoder(
    tf=None,
    latent_dim: int = 16,
    conv: bool = False,
    base_filters: int = 32,
    hidden_units: int = 256,
) -> Autoencoder:
    """Crea un :class:`Autoencoder`. ``tf`` se mantiene por compatibilidad."""
    return Autoencoder(latent_dim=latent_dim, conv=conv, base_filters=base_filters, hidden_units=hidden_units)


def build_encoder_decoder(
    tf=None,
    latent_dim: int = 2,
    conv: bool = False,
    base_filters: int = 32,
    hidden_units: int = 256,
):
    """Devuelve el encoder variacional y el decoder del VAE."""
    encoder = build_variational_encoder(latent_dim, conv, base_filters, hidden_units)
    decoder = build_decoder(latent_dim, conv, base_filters, hidden_units)
    return encoder, decoder


def build_vae(
    tf=None,
    latent_dim: int = 2,
    conv: bool = False,
    base_filters: int = 32,
    hidden_units: int = 256,
    kl_weight: float = 1.0,
) -> VariationalAutoencoder:
    """Crea un :class:`VariationalAutoencoder` listo para entrenar."""
    return VariationalAutoencoder(
        latent_dim=latent_dim,
        conv=conv,
        base_filters=base_filters,
        hidden_units=hidden_units,
        kl_weight=kl_weight,
    )
