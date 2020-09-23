from flask import Flask, request, send_file
from flask_restful import Resource, Api
from get_similar_questions import get_similar_question


app = Flask(__name__)
api = Api(app)


class Quora(Resource):
    def get(self):
        return {"test": "Hello"}
        
    def post(self):
        data = request.get_json(force=True)
        print("JSON Received!")
        question = data['question']

        print(question)

        return get_similar_question(question)
        
api.add_resource(Quora, '/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True, use_reloader=True)