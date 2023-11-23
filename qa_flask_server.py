from flask import Flask, request, jsonify
from flask_cors import CORS  
from transformers import pipeline

app = Flask(__name__)
CORS(app)

model_name = "skandavivek2/roberta-finetuned-subjqa-movies_2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    if 'context' not in data or 'question' not in data:
        return jsonify({'error': 'Please provide both context and question.'}), 400
    
    context = data['context']
    question = data['question']

    try:
        result = nlp({'question': question, 'context': context})
        answer = result['answer']
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
