from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import requests
import uuid

app = Flask(__name__)
CORS(app)

# Load YOLO model once on startup (for better performance)
model = YOLO('yolo_weights/best.pt').to('cpu')

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
    return jsonify({"status": "ok", "message": "Mahjong backend is running! POST an image to /analyze."})

@app.route('/analyze', methods=['POST'])
def analyze_hand():
    try:
        print("Request received.")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        unique_id = str(uuid.uuid4())
        temp_filename = f"temp_{unique_id}.jpg"
        predict_name = f"predict_{unique_id}"

        try:
            # Save image
            image.save(temp_filename)
            print(f"Image saved to {temp_filename}")

            # Run YOLO detection
            print("Running YOLO...")
            results = model(temp_filename, save=False, save_txt=True, save_conf=True,
                            project='yolo_output', name=predict_name)
            print("YOLO done.")

            # Parse label output
            label_path = os.path.join('yolo_output', predict_name, 'labels', f'temp_{unique_id}.txt')
            tile_vector = parse_yolo_txt(label_path)

            if not tile_vector or tile_vector == ["No tiles detected"]:
                return jsonify({
                    "tiles": [],
                    "suggestion": "No mahjong tiles were detected in the image. Please ensure the image is clear and contains visible mahjong tiles."
                })

            # Call Gemini
            prompt = f"Given this Mahjong hand (Singapore mahjong rules): {', '.join(tile_vector)}, suggest the best tile to discard and explain why."
            gemini_response = call_gemini_api(prompt)

            return jsonify({
                "tiles": tile_vector,
                "suggestion": gemini_response,
                "status": "success"
            })

        except Exception as e:
            print(f"YOLO or image processing error: {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                print(f"Temp file {temp_filename} cleaned up.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def parse_yolo_txt(filepath):
    tiles = []
    try:
        if not os.path.exists(filepath):
            print(f"Label file not found: {filepath}")
            return ["No tiles detected"]

        with open(filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("Label file is empty")
            return ["No tiles detected"]

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(tile_classes):
                        tile_name = tile_classes[class_id]
                        tiles.append(tile_name)
                        print(f"Detected tile: {tile_name}")
                    else:
                        print(f"Invalid class_id: {class_id}")
                except ValueError as e:
                    print(f"Parsing error: {e}")

    except Exception as e:
        print(f"Error parsing label file: {e}")
        return ["No tiles detected"]

    return tiles if tiles else ["No tiles detected"]

def call_gemini_api(prompt):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        json_data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=json_data,
            timeout=30
        )

        print(f"Gemini response code: {response.status_code}")
        if response.status_code != 200:
            return f"Gemini API error: {response.status_code} - {response.text}"

        response_data = response.json()
        if 'candidates' in response_data and response_data['candidates']:
            parts = response_data['candidates'][0].get('content', {}).get('parts', [])
            return parts[0]['text'] if parts else "No response from Gemini."

        return "Unexpected Gemini API format."

    except requests.exceptions.Timeout:
        return "Gemini API request timed out."
    except Exception as e:
        print(f"Gemini exception: {e}")
        return f"Gemini API error: {str(e)}"

if __name__ == '__main__':
    os.makedirs('yolo_output', exist_ok=True)
    print("MahjongLah backend running...")
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
