from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, support_credentials=True)

db = ["asdf", "asdf"]

@app.route('/')
def home():
    return jsonify(data=db)

@app.route('/rpi', methods=['GET', 'POST'])
def rpi():
    global db

    device = request.get_json()['device']
    command = request.get_json()['command']
    db.append(f'{device}: {command}')
    return jsonify()