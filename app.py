from flask import Flask, request, jsonify, render_template
from src.rag_pipeline import RAGChatbot
from src.vector_store import add_chunks_to_db, get_all_documents, delete_document
from src.document_processor import process_document
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["50 per hour", "10 per minute"]
)

bot = RAGChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    data = request.json
    user_message = data.get('message')
    source_filter = data.get('source_filter')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    if len(user_message) > 2000:
        return jsonify({'error': 'Message too long. Max 2000 characters.'}), 400

    try:
        response = bot.chat(user_message, source_filter=source_filter)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Sorry, I encountered an error: {str(e)}. Please try again.'}), 200

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per hour")
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    if len(file.filename) > 100:
        return jsonify({'error': 'Filename too long'}), 400

    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    chunks = process_document(file_path)
    add_chunks_to_db(chunks, file.filename)

    return jsonify({
        'message': f'Successfully processed {file.filename}',
        'chunks': len(chunks)
    })

@app.route('/documents', methods=['GET'])
def documents():
    docs = get_all_documents()
    return jsonify({'documents': docs})

@app.route('/delete', methods=['POST'])
def delete():
    data = request.json
    source_name = data.get('source_name')
    delete_document(source_name)
    return jsonify({'message': f'Deleted {source_name}'})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)