import requests
from termcolor import colored

if __name__ == '__main__':
    url = "http://prophet0097.natapp1.cc/wh"
    # url = "http://127.0.0.1:8003/wh"
    data = colored("hahaha", 'red')
    res = requests.post(url, data=data)

