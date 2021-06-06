# Required imports
import os
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app

# Get model text generation function
from gentext import gentext

# Initialize Flask app
app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('todos')

@app.route('/gentext', methods=['POST'])
def gentxt():
    """
        gentxt() : Return primary Murakami text from starting
        message from request body. String 'msg' must be included
        in request body (e.g. json={... msg: 'sample_string'}).
    """
    try:
        msg = request.json['msg']
        to_send = gentext(msg)
        return jsonify({"success": True, "gentxt": to_send}), 200
    except Exception as e:
        return f"An Error Occured: {e}"

port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port)