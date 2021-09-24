import requests
from flask import Flask, request

requests.adapters.DEFAULT_RETRIES = 5
app = Flask(__name__)


@app.route('/wh', methods=['POST'])
def respond():
    resp = request.get_data()
    print(resp)

    return ""


if __name__ == '__main__':
    app.run(debug=False, port=8003, host='127.0.0.1')
