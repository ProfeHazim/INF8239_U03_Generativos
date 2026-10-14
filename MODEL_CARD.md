# Model Card · Autoencoder y VAE

## Modelos y versiones

| Modelo | Clase | Latente | Parámetros | Formato |
| --- | --- | --- | --- | --- |
| Autoencoder | `Autoencoder` (`conv=False`, denso) | `z in R^16` determinista | 410,912 | `models/autoencoder.keras` |
| Decoder del VAE | `VariationalAutoencoder.decoder` (`conv=False`, denso) | `z in R^2` gaussiano | 202,256 | `models/decoder.keras` |

- TensorFlow 2.21.0 · Keras 3.15.1 · Python 3.12.14.
- Optimizador Adam, tamaño de lote 256, salidas con activación sigmoide.
- Reproducible con `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8`
  (semilla fija `SEED = 42`).

## Uso previsto y usuarios

- Demostración académica para la unidad INF-8239 (modelos generativos).
- Exploración del espacio latente del VAE (2D) mediante la app `app/streamlit_app.py`.
- Referencia docente para comparar un AE determinista con un VAE probabilístico.

## Usos fuera de alcance

- **No** debe usarse como generador de prendas reales ni para decisiones de
  diseño, inventario o producción.
- **No** es un modelo de privacidad: generar datos sintéticos no garantiza
  anonimato ni elimina el sesgo del dataset original.
- **No** usar como componente de un sistema biométrico o de clasificación en
  producción sin reentrenamiento y validación específicos.

## Dataset, licencia y particiones

- Dataset: **Fashion-MNIST** (Zalando Research), 10 clases de prendas, imágenes
  en escala de grises de 28x28 píxeles.
- Licencia: **MIT**.
- Particiones: 60,000 muestras de entrenamiento y 10,000 de prueba. El AE usa
  además un 10% del entrenamiento como validación (`validation_split=0.1`).
- Preprocesamiento: normalización a `[0, 1]` con `normalize_images` y validación
  de forma/rango con `validate_images`.

## Arquitectura y espacio latente

Ambas arquitecturas procesan imágenes de 28x28x1 y comparten los bloques de
codificación/decodificación (`conv=False` denso, `conv=True` convolucional).

### Autoencoder (AE) — espacio latente determinista

- Encoder `x -> z` con `z in R^16` (configurable con `latent_dim`), una sola
  dimensión por variable latente. `build_autoencoder(tf, latent_dim=16)`.
- Decoder `z -> x_hat` con salida sigmoide en `[0, 1]`.
- Pérdida: solo reconstrucción (BCE). No modela una densidad, por lo que el
  espacio latente no es continuo ni muestreable de forma natural.

### Variational Autoencoder (VAE) — espacio latente probabilístico

- Encoder variacional `q(z|x) = N(z_mean, diag(exp(z_log_var)))`, por defecto
  `latent_dim=2` (para visualizar/explorar el plano latente).
- Prior `p(z) = N(0, I)`.
- *Reparameterization Trick*: `z = z_mean + exp(0.5 * z_log_var) * eps`, con
  `eps ~ N(0, I)`, lo que hace diferenciable el muestreo.
- Pérdida combinada: `L = L_rec + kl_weight * KL(q(z|x) || p(z))`, con
  `KL = -0.5 * sum(1 + z_log_var - z_mean^2 - exp(z_log_var))`.

## Métricas de reconstrucción y generación

Corrida de referencia (`--epochs-ae 6 --epochs-vae 8`, lote 256):

| Modelo | Época | Pérdida | Reconstrucción (BCE) | KL |
| --- | --- | --- | --- | --- |
| AE | 6 | val_loss ≈ 0.2857 | — | — |
| VAE | 8 | ≈ 270.78 | ≈ 264.56 | ≈ 6.22 |

- La pérdida del AE es BCE promediada; la del VAE suma la reconstrucción sobre
  los píxeles más la KL, por lo que las escalas **no son comparables** entre
  modelos.
- La KL desciende de ≈ 10.6 (época 1) a ≈ 6.2 (época 8): el encoder se acerca a
  la prior `N(0, I)` sin colapsar (no se observa *posterior collapse*).
- Figuras: `reports/ae_vae_panel.png` (originales, reconstrucción AE,
  reconstrucción VAE y muestras del prior) y `reports/interpolation.png`.

## Diversidad y vecinos cercanos

- El panel (`ae_vae_panel.png`) permite comparar visualmente la fidelidad del AE
  y del VAE frente a las imágenes originales y frente a muestras sintéticas.
- La rejilla de `interpolation.png` recorre el plano latente 2D y muestra
  transiciones suaves, evidencia de que el VAE aprendió un espacio latente
  continuo; el AE no ofrece esta propiedad.
- Al ser el latente del VAE de solo 2 dimensiones, su capacidad es limitada y
  las muestras son necesariamente difusas.

## Hardware, tiempo y tamaño

- Hardware: CPU (Windows nativo, sin GPU; TensorFlow ≥ 2.11 no usa GPU en
  Windows nativo). Tiempos medidos solo como referencia.

| Modelo | Épocas | Tiempo de entrenamiento |
| --- | --- | --- |
| AE | 6 | ≈ 13.6 s |
| VAE | 8 | ≈ 23.0 s |

| Artefacto | Tamaño |
| --- | --- |
| `models/autoencoder.keras` (410,912 parámetros) | ≈ 1.6 MB |
| `models/decoder.keras` (202,256 parámetros) | ≈ 0.8 MB |

## Límites del VAE: nitidez y sesgo

### Nitidez de la reconstrucción

La reconstrucción del VAE **no es perfecta ni completamente nítida**. La causa
principal es la función de pérdida de reconstrucción: una **entropía cruzada
binaria (BCE) aplicada píxel a píxel e independiente**, que mide el error de
cada píxel por separado. Como no modela correlaciones espaciales ni estructura
global, el óptimo tiende a la media por píxel y produce salidas **borrosas**:

- Detalles finos (texturas, bordes, patrones como el estampado a cuadros) se
  pierden.
- El término KL y el latente 2D limitan además cuánta información puede retener
  el cuello de botella, lo que refuerza el efecto de suavizado.
- Alternativas que mejoran la nitidez (fuera del alcance de este ejercicio):
  pérdida MSE/L1, pérdidas perceptuales, o decoders convolucionales más
  profundos (`conv=True`).

### Sesgo de los datos

El modelo aprende **únicamente** de Fashion-MNIST: **toda la ropa generada
refleja exclusivamente ese dataset**. En consecuencia:

- Solo aparecen las **10 categorías** del dataset (camisetas, pantalones,
  jerséis, vestidos, abrigos, sandalias, camisas, zapatillas, bolsos y botines),
  con su estética, resolución (28x28 gris), encuadre y estilo típicos.
- No hay diversidad de tallas, colores, culturas, géneros ni prendas modernas
  ausentes del dataset; los sesgos presentes en Fashion-MNIST se heredan y
  pueden amplificarse.
- Las muestras sintéticas **no representan** el mundo real ni una población
  diversa: son interpolaciones/muestreos del prior aprendido sobre esta
  distribución concreta.

## Limitaciones, riesgos y supervisión

Ver arriba "Límites del VAE: nitidez y sesgo" para el detalle de borrosidad y
sesgo. Adicionalmente:

- El AE es determinista y no modela densidad: no permite muestrear de forma
  natural; solo decodifica puntos arbitrarios del latente.
- Los resultados dependen de las pocas épocas y del latente reducido (2D para el
  VAE); más capacidad/épocas mejorarían la calidad pero no eliminan el sesgo.
- Supervisión humana obligatoria: cualquier uso debe auditarse contra el sesgo
  del dataset y **no** debe afirmarse privacidad por generar datos sintéticos.
