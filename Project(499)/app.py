from flask import Flask, render_template, request
from transformers import pipeline
import os
from werkzeug.utils import secure_filename
import PyPDF2

app = Flask(__name__,static_url_path='/static', static_folder='static')

# Initialize the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=["POST"])  
def summarize():

    if 'pdf' not in request.files:
        return "No file part"

    pdf = request.files['pdf']

    if pdf.filename == '':
        return "No selected file"

    if pdf:
        filename = secure_filename(pdf.filename)
        pdf.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Extract text from the PDF using PdfReader
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()

        # Use the summarization model to generate a summary
        summary = summarizer(
            pdf_text,
            max_length=512,
            min_length=50,
            do_sample=False
            )

        return render_template("result.html", summary=summary[0]['summary_text'])

if __name__ == '__main__':
    app.run(debug=True)
