import os

from flask import Flask, request, render_template
import tensorflow as tf

from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = "staic/uploads"

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "lion_or_cheetah.keras"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/classify/")
def upload_file():
    file= request.files["image"]
    upload_image_url = os.path.join(UPLOAD_FOLDER, file.name)
    file.save(upload_image_url)
    label, prob = classify(cnn_model, upload_image_url)
    return render_template("result.html", label=label, prob=prob)


if __name__ == "__main__":
    app.run(debug=True)
