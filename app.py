from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-prepend")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-prepend")

# Function to generate questions
def generate_questions_from_text(text, max_questions=5):
    sentences = sent_tokenize(text)
    sentences = sentences[:max_questions]

    questions = []
    for sentence in sentences:
        input_text = f"generate question: {sentence}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        outputs = model.generate(
            input_ids,
            max_length=50,
            num_beams=8,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions.append(question)

    return questions

@app.route("/", methods=["GET", "POST"])
def index():
    questions = []
    if request.method == "POST":
        text = request.form["text"]
        if text:
            questions = generate_questions_from_text(text)
    return render_template("index.html", questions=questions)

if __name__ == "__main__":
    app.run(debug=True)

