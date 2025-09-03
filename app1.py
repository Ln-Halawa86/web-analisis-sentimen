from flask import Flask, render_template, request, redirect, make_response, jsonify
import re
from sentiment_labeler import SentimentLabeler

app = Flask(__name__)

try:
    labeler = SentimentLabeler()
except Exception as e:
    print(f"Error: {e}")
    labeler = None

# Fungsi untuk memuat kamus normalisasi
def load_normalization_dict(filepath):
    norm_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                slang, formal = line.strip().split('\t')
                norm_dict[slang] = formal
    return norm_dict

# Fungsi case folding dan cleansing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenize
def tokenize(text):
    return text.split()

# Normalisasi
def normalize_tokens(tokens, norm_dict):
    return [norm_dict.get(token, token) for token in tokens]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     norm_dict = load_normalization_dict('kamus/normalisasi.txt')
#     history = request.cookies.get('history', '').split('|') if request.cookies.get('history') else []
    
#     original_text = ""
#     normalized_text = ""
    
#     if request.method == 'POST':
#         original_text = request.form['text']
#         cleaned = preprocess(original_text)
#         tokens = tokenize(cleaned)
#         normalized_tokens = normalize_tokens(tokens, norm_dict)
#         normalized_text = ' '.join(normalized_tokens)

#         # Simpan ke history
#         new_entry = f"{original_text} → {normalized_text}"
#         history.insert(0, new_entry)
#         history = history[:5]  # Simpan maksimal 5 riwayat terakhir

#     resp = make_response(render_template('coba.html',
#                                          original=original_text,
#                                          result=normalized_text,
#                                          history=history))
    
#     if history:
#         resp.set_cookie('history', '|'.join(history), max_age=60*60*24*30)  # 30 hari
    
#     return resp

@app.route('/labeling-otomatis', methods=['GET', 'POST'])
def auto_labeling():
    if not labeler:
        return "Gagal memuat leksikon. Periksa file bobot kata.", 500

    result = None
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            label, score, matches = labeler.label_text(text)
            result = {
                'text': text,
                'label': label,
                'score': round(score, 2),
                'matches': matches  # kata yang cocok beserta bobot dan jumlah
            }
    return render_template('auto_labeling.html', result=result)

if __name__ == '__main__':
    app.run(port=5002, debug=True)