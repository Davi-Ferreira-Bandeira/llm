from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import requests, sqlite3, os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
DB = "blog.db"

# Criar tabelas
with sqlite3.connect(DB) as con:
    con.execute("""CREATE TABLE IF NOT EXISTS posts
                   (id INTEGER PRIMARY KEY, tema TEXT, conteudo TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS comentarios
                   (id INTEGER PRIMARY KEY, post_id INT, texto TEXT,
                    sentimento TEXT, confianca REAL)""")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_API = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def gerar_post_com_llm(tema):
    h = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }
    m = [
        {"role": "system", "content": "Escreva posts curtos e envolventes em português."},
        {"role": "user", "content": f"Escreva um post sobre '{tema}'."
    ]
    p = {
        "model": MODEL,
        "messages": m,
        "max_tokens": 250,
        "temperature": 0.8
    }
    r = requests.post(HF_API, headers=h, json=p).json()
    return r["choices"][0]["message"]["content"].strip()

analisador = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    device=-1  # força uso da CPU
)

def analisar_sentimento(txt):
    r = analisador(txt)[0]
    mapa = {
        'LABEL_0': 'Negativo',
        'LABEL_1': 'Neutro',
        'LABEL_2': 'Positivo'
    }
    return mapa.get(r['label'], r['label']), r['score'] * 100

@app.route('/')
def index():
    with sqlite3.connect(DB) as con:
        posts = con.execute(
            "SELECT id, tema, conteudo FROM posts ORDER BY id DESC"
        ).fetchall()
        comentarios = {
            p[0]: con.execute(
                "SELECT texto, sentimento, confianca FROM comentarios WHERE post_id=?",
                (p[0],)
            ).fetchall()
            for p in posts
        }
    return render_template('index.html', posts=posts, comentarios=comentarios)

@app.route('/gerar-post', methods=['POST'])
def gerar_post():
    tema = request.json.get('tema', '')
    if not tema:
        return jsonify({'erro': 'tema vazio'}), 400
    conteudo = gerar_post_com_llm(tema)
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO posts(tema, conteudo) VALUES(?, ?)", 
                    (tema, conteudo))
    return jsonify({'post': conteudo})

@app.route('/comentar', methods=['POST'])
def comentar():
    data = request.json
    texto = data.get('comentario', '')
    post_id = data.get('post_id')
    if not texto or not post_id:
        return jsonify({'erro': 'dados inválidos'}), 400
    sent, conf = analisar_sentimento(texto)
    with sqlite3.connect(DB) as con:
        con.execute(
            "INSERT INTO comentarios(post_id, texto, sentimento, confianca) VALUES(?, ?, ?, ?)",
            (post_id, texto, sent, conf)
        )
    return jsonify({'sentimento': sent, 'confianca': conf})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)