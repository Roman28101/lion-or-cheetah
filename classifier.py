from typing import Any

import tensorflow as tf


IMAGE_SIZE = (224, 224)


def process_image(image: tf.Tensor) -> tf.Tensor:
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def load_and_preprocess_image(path: str) -> tf.Tensor:
    image = tf.keras.preprocessing.image.load_img(
        path, target_size=IMAGE_SIZE
    )

    return process_image(image)


def classify(model: tf.keras.Model, image_path: str) -> tuple[str, int | Any]:
    preprocessed_image = load_and_preprocess_image(image_path)

    predictions = model.predict(preprocessed_image)
    score = predictions[0][0]

    label = "lion" if score <= 0.5 else "cheetah"
    prob = 1 - score if label == "lion" else score

    return label, prob
