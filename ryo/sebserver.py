from flask import Flask, render_template, request, jsonify
from ryo.make_samples import MakeSamples
from ryo.linear_regression import LinearRegression

app = Flask(__name__)

ms = MakeSamples()
lr = LinearRegression()
ms.to_csv()
lr.do_train()


@app.route('/')
def echo():
    return render_template('sample.html')


@app.route('/api', methods=['POST'])
def get_prediction():
    data = request.get_json()
    return str(lr.predict(data['x']))


app.run(port=8080, debug=True)
