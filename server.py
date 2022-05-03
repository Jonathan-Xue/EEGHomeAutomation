from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, support_credentials=True)

db = []

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)