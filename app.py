from flask import Flask, render_template, request, jsonify
import openai
import pdfplumber
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = Flask(__name__)

# Directly including the OpenAI API key (not recommended for production)
openai.api_key = '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'

# Default PDF file path
DEFAULT_PDF_FILE = r"C:\Users\haris\Downloads\rules short.pdf"


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    with pdfplumber.open(pdf_file) as pdf:
        text = "".join(page.extract_text()
                       for page in pdf.pages if page.extract_text())
    return text


def summarize_text(text, ratio=0.5):
    """Summarizes the extracted text."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=int(
        ratio * len(parser.document.sentences)))
    return " ".join(str(sentence) for sentence in summary)


def answer_question(question, context, max_tokens=150):
    """Generates an answer to the question based on the provided context."""
    shortened_context = context[-2048:]  # Using the most relevant part of the context
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Context: {shortened_context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response.choices[0].text.strip()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            question = request.form['question']
            pdf_text = extract_text_from_pdf(DEFAULT_PDF_FILE)
            # pdf_text = summarize_text(pdf_text, ratio=0.3)  # Optional summarization

            answer = answer_question(question, pdf_text)
            return jsonify({'question': question, 'answer': answer})
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
