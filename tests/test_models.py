import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import (
    Autoencoder,
    VariationalAutoencoder,
    build_autoencoder,
    build_encoder_decoder,
)

BATCH = np.zeros((2, 28, 28, 1), dtype="float32")


def test_autoencoder_output_contract():
    model = build_autoencoder(tf)
    result = model(BATCH).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_decoder_output_contract():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    result = decoder(np.zeros((2, 2), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_autoencoder_latent_space_is_deterministic():
    model = build_autoencoder(tf, latent_dim=5)
    assert isinstance(model, Autoencoder)
    assert model.latent_shape == (5,)
    codes = model.encode(np.random.rand(2, 28, 28, 1).astype("float32")).numpy()
    assert codes.shape == (2, 5)


def test_vae_encoder_defines_gaussian_latent_space():
    vae = VariationalAutoencoder(latent_dim=3)
    assert vae.latent_shape == (3,)
    mean, log_variance = vae.encode(BATCH)
    assert mean.shape == (2, 3)
    assert log_variance.shape == (2, 3)


def test_reparameterization_trick_shape_and_zero_noise_mean():
    tf.random.set_seed(0)
    vae = VariationalAutoencoder(latent_dim=4)
    latent = vae.reparameterize(tf.zeros((512, 4)), tf.zeros((512, 4)))
    assert latent.shape == (512, 4)
    # Con mu=0 y log_var=0 -> z ~ N(0, I), la media empírica debe ser ~0.
    assert abs(float(tf.reduce_mean(latent))) < 0.2


def test_kl_divergence_is_zero_for_standard_normal():
    vae = VariationalAutoencoder(latent_dim=3)
    kl = vae.kl_divergence(tf.zeros((4, 3)), tf.zeros((4, 3))).numpy()
    assert np.isclose(kl, 0.0, atol=1e-6)


def test_vae_reconstruction_contract_and_combined_loss_terms():
    vae = VariationalAutoencoder(latent_dim=2)
    reconstruction = vae(BATCH).numpy()
    assert reconstruction.shape == (2, 28, 28, 1)
    mean, log_variance = vae.encode(BATCH)
    reconstruction_loss = vae.reconstruction_loss(BATCH, reconstruction).numpy()
    kl_loss = vae.kl_divergence(mean, log_variance).numpy()
    assert reconstruction_loss >= 0.0
    assert kl_loss >= 0.0


def test_vae_sampling_uses_prior():
    vae = VariationalAutoencoder(latent_dim=2)
    assert vae.sample(5, seed=0).numpy().shape == (5, 28, 28, 1)
