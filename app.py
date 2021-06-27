from flask import Flask, render_template, request
from predict import *
import numpy as np

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        text_ps = request.form['personal statement']
        text_cv = request.form['cv']
        predict_ps = predictPS(text_ps)
        predict_cv = predictCV(text_cv)
        print(predict_ps, "\n", predict_cv)

        input = [[predict_ps, predict_cv[0], predict_cv[1], predict_cv[2], predict_cv[3]]]

        output = getPrediction(np.asarray(input))
        print(output)
        return render_template('result.html', classification=output)


if __name__ == "__main__":
    app.run(debug=True)
