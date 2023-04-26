from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2Model

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def get_answer():
    question = request.form['question']
    encoded_input = tokenizer(question, return_tensors='pt')
    output = model(**encoded_input)
    answer = tokenizer.decode(output['logits'][0], skip_special_tokens=True)
    return render_template('index.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
