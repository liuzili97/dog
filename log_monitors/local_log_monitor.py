import requests
from flask import Flask, request

requests.adapters.DEFAULT_RETRIES = 5
app = Flask(__name__)


@app.route('/wh', methods=['POST'])
def respond():
    resp = request.get_data()
    print("\n\n")
    print(str(resp, 'utf-8'))
    print("\n\n")

    return ""


if __name__ == '__main__':
    app.run(debug=False, port=8004, host='127.0.0.1')
