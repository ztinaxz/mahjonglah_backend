from flask import Flask, request, jsonify
import subprocess
import os
import requests

app = Flask(__name__)

# List of your classes in YOLO order
tile_classes = [
    'animal-cat', 'animal-centipede', 'animal-mouse', 'animal-rooster',
    'bamboo-1', 'bamboo-2', 'bamboo-3', 'bamboo-4', 'bamboo-5', 'bamboo-6',
    'bamboo-7', 'bamboo-8', 'bamboo-9',
    'bonus-autumn', 'bonus-bamboo', 'bonus-chrysanthemum', 'bonus-orchid',
    'bonus-plum', 'bonus-spring', 'bonus-summer', 'bonus-winter',
    'characters-1', 'characters-2', 'characters-3', 'characters-4', 'characters-5',
    'characters-6', 'characters-7', 'characters-8', 'characters-9',
    'dots-1', 'dots-2', 'dots-3', 'dots-4', 'dots-5', 'dots-6', 'dots-7', 'dots-8', 'dots-9',
    'honors-east', 'honors-green', 'honors-north', 'honors-red', 'honors-south', 'honors-west', 'honors-white'
]

@app.route('/')
def home():
    return "Mahjong backend is running! POST an image to /analyze."

@app.route('/analyze', methods=['POST'])
def analyze_hand():
    image = request.files['image']
    image.save('temp.jpg')
    
    subprocess.run([
        'yolo', 'detect', 'predict',
        'model=yolo_weights/best.pt',
        'source=temp.jpg',
        'project=yolo_output',   
        'name=predict',
        'save_txt=True',
        'save_conf=True'
    ], check=True)

    # 🔎 Find the latest YOLO output folder (predict7, predict8, etc.)
    latest_run = sorted([d for d in os.listdir('yolo_output') if d.startswith('predict')])[-1]
    label_path = os.path.join('yolo_output', latest_run, 'labels', 'temp.txt')

    # ✅ Build your tile vector from label file
    tile_vector = parse_yolo_txt(label_path)
    print(f"Tiles detected: {tile_vector}")  # ← This prints to terminal for debugging

    # ✅ Turn vector into prompt string
    prompt = f"Given this Mahjong hand (Singapore mahjong rules): {', '.join(tile_vector)}, suggest the best tile to discard and explain why."

    # ✅ Send prompt to Gemini
    gemini_response = call_gemini_api(prompt)
    
    return jsonify({"tiles": tile_vector, "suggestion": gemini_response})

def parse_yolo_txt(filepath):
    tiles = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                tile_name = tile_classes[class_id]
                tiles.append(tile_name)
    except FileNotFoundError:
        return ["No tiles detected"]
    return tiles


def call_gemini_api(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not set."

    # Use a known working model from AI Studio:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    json_data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=json_data
    )
    response.raise_for_status()
    return response.json()['candidates'][0]['content']['parts'][0]['text']

if __name__ == '__main__':
    app.run(debug=True)
