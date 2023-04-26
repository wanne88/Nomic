from flask import Flask, request, jsonify
from nomic.gpt4all import GPT4All

app = Flask(__name__)
m = GPT4All()
m.open()

@app.route("/", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    question = data["question"]
    answer = m.prompt(question)
    return jsonify(answer=answer)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
