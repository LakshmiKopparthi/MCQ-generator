from flask import Flask, request, jsonify
from MCQ import generate_mcqs  # Import MCQ function

app = Flask(__name__)

@app.route('/')
def home():
    return "MCQ Generator API is running!"

@app.route('/generate_mcqs', methods=['POST'])
def mcq_endpoint():
    try:
        data = request.json
        num_questions = int(data.get('num_questions', 5))

        if num_questions < 1:
            return jsonify({"status": "error", "message": "Number of questions must be at least 1"}), 400

        mcqs = generate_mcqs(num_questions)

        return jsonify({"status": "success", "mcqs": mcqs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

