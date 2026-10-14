# Declaración de uso de inteligencia artificial

## Herramientas utilizadas

- **Modelo de lenguaje Gemini 3.1 Pro**, utilizado como consultor técnico interactivo (chatbot) para asistencia en depuración, revisión de sintaxis y consultas teóricas durante el desarrollo de la Unidad 03 (INF-8239).
- Documentación oficial de TensorFlow 2.21 / Keras 3.15 y de scikit-learn como fuentes principales de referencia y validación.

## Propósito de cada uso

1. **Consultas para el diseño de arquitecturas (LAB10 / Ejercicio 06).** Se utilizó Gemini para discutir y validar la transición de arquitecturas funcionales a la API de clases orientada a objetos de Keras. Específicamente, se consultó cómo programar correctamente el *Reparameterization Trick* (`z = z_mean + exp(0.5 * z_log_var) * eps`) y la fusión de la pérdida combinada (reconstrucción BCE + divergencia KL) asegurando que el flujo de gradientes se mantuviera intacto.
2. **Asistencia en depuración de scikit-learn (LAB09).** Durante el desarrollo híbrido, surgió un error de compatibilidad de tipos. Se proporcionó el *traceback* a Gemini, lo cual ayudó a identificar rápidamente que las versiones recientes de scikit-learn ya no soportan la clase `numpy.matrix`. Con este diagnóstico, rediseñé la lógica para convertir las matrices a `numpy.ndarray` estándar, solucionando el fallo.
3. **Investigación sobre cambios de versión en Keras 3 (LAB10).** Al notar un error en la reducción de tensores durante el entrenamiento del VAE, consulté a Gemini sobre posibles cambios en la función `tf.keras.losses.binary_crossentropy`. La IA me ayudó a descubrir que en Keras 3 esta función reduce por defecto el eje del canal (`axis=-1`). Con este dato, modifiqué manualmente el código para sumar únicamente las dimensiones espaciales (`reduce_sum(..., axis=(1, 2))`).
4. **Revisión editorial y formateo.** Se utilizó la IA para mejorar la coherencia gramatical del análisis de resultados, estructurar las tablas comparativas en formato Markdown y traducir técnicamente los mensajes de control de versiones (*commits*).

## Prompts y decisiones relevantes

- Se diseñaron prompts específicos consultando: *"¿Cómo estructurar de manera limpia un Variational Autoencoder usando `tf.keras.Model` definiendo explícitamente el espacio latente y manteniendo compatibilidad funcional?"*.
- Se consultó a la IA sobre las mejores prácticas para hacer los modelos serializables. Tras analizar las respuestas, tomé la decisión de usar modelos funcionales (`Autoencoder.reconstruction_model` y el decoder del VAE) para que Streamlit pudiera cargarlos directamente con `tf.keras.models.load_model` sin necesidad de dependencias complejas.
- Se pidió a Gemini resumir las implicaciones teóricas de la divergencia KL en espacios latentes, lo cual sirvió de inspiración para redactar el análisis crítico del `MODEL_CARD.md`.

## Elementos verificados por el equipo

Todas las sugerencias proporcionadas por la IA fueron **validadas técnicamente e implementadas de forma manual** por mi persona antes de su integración definitiva:

- **Matemática y Arquitecturas:** Revisión exhaustiva de la lógica del truco de reparametrización. Se ejecutó la suite de pruebas unitarias locales (`uv run pytest -q`, **10 passed**) y se verificó el cumplimiento de estándares PEP-8 con el linter (`uv run ruff check`).
- **Verificación de Shapes (Keras 3):** Se comprobó empíricamente mediante impresiones en consola (`print(tf.shape)`) que la salida de la pérdida de reconstrucción tuviera la reducción espacial correcta tras el ajuste de ejes.
- **Pipeline Completo:** Se confirmó que la generación de datos y las recargas de los modelos (`models/autoencoder.keras`, `models/decoder.keras`) fueran funcionales y reproducibles fijando la semilla (`SEED = 42`). Se evaluaron visualmente los paneles resultantes en la carpeta `reports/`.

## Cambios realizados sobre los resultados generados por IA

- El código sugerido por Gemini **nunca se insertó como copy-paste**. Se utilizó como pseudocódigo o referencia teórica, el cual transcribí y adapté a la estructura, estilo y nomenclatura del repositorio (añadiendo *docstrings* en español, variables claras y control de flujos).
- Se descartaron sugerencias genéricas de la IA que no se alineaban con las restricciones del mandato del profesor (como el aislamiento estricto de TensorFlow).

## Responsabilidad asumida

Asumo la responsabilidad total y absoluta sobre el diseño, el código final integrado, los modelos entrenados, las métricas extraídas y las conclusiones reportadas. La Inteligencia Artificial se empleó estrictamente como herramienta de consulta (brainstorming y diagnóstico) y **no** como sustituto de mi capacidad analítica ni de la validación técnica. Se deja constancia de que la generación de datos sintéticos **no** garantiza privacidad ni elimina el sesgo inherente del dataset Fashion-MNIST.
