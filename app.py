from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model (IMPORTANT)
model = SentenceTransformer('all-MiniLM-L6-v2')  
# If you saved model → use: SentenceTransformer("model")

# Your dataset (replace later if needed)
recipes = ["Masala dosa", "Idli", "Paneer butter masala", "Upma"]

recipe_embeddings = model.encode(recipes)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data['query']

    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, recipe_embeddings)

    best_index = scores.argmax()
    result = recipes[best_index]

    return jsonify({"result": result})

@app.route('/')
def home():
    return "API is running"

if __name__ == '__main__':
    app.run(debug=True)