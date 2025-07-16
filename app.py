from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import requests
import tempfile
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Enable CORS for React Native requests
CORS(app)

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

@app.route('/test-upload', methods=['POST'])
def test_upload():
    """Simple endpoint to test file upload"""
    try:
        print("=== TEST UPLOAD DEBUG ===")
        print(f"Request method: {request.method}")
        print(f"Request content type: {request.content_type}")
        print(f"Request files keys: {list(request.files.keys())}")
        print(f"Request form keys: {list(request.form.keys())}")
        print(f"Request data length: {len(request.data)}")
        
        if request.files:
            for key, file in request.files.items():
                print(f"File key: {key}, filename: {file.filename}, content_type: {file.content_type}")
        
        return jsonify({
            "status": "received",
            "files": list(request.files.keys()),
            "form": list(request.form.keys()),
            "content_type": request.content_type
        })
    except Exception as e:
        print(f"Test upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_hand():
    try:
        # Debug: Print all request info
        print(f"Request method: {request.method}")
        print(f"Request content type: {request.content_type}")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request form: {list(request.form.keys())}")
        print(f"Request headers: {dict(request.headers)}")
        
        # Check if image was uploaded
        if 'image' not in request.files:
            print("ERROR: No 'image' key in request.files")
            return jsonify({"error": "No image file provided"}), 400
        
        image = request.files['image']
        print(f"Image filename: {image.filename}")
        print(f"Image content type: {image.content_type}")
        
        if image.filename == '':
            print("ERROR: Empty filename")
            return jsonify({"error": "No image file selected"}), 400
        
        # Generate unique filename to avoid conflicts
        unique_id = str(uuid.uuid4())
        temp_filename = f"temp_{unique_id}.jpg"
        
        try:
            # Save uploaded image
            image.save(temp_filename)
            print(f"Image saved as: {temp_filename}")
            
            # Create unique output directory name
            predict_name = f"predict_{unique_id}"
            
            # Run YOLO detection
            print("Running YOLO detection...")
            result = subprocess.run([
                'yolo', 'detect', 'predict',
                f'model=yolo_weights/best.pt',
                f'source={temp_filename}',
                'project=yolo_output',
                f'name={predict_name}',
                'save_txt=True',
                'save_conf=True'
            ], capture_output=True, text=True, check=True)
            
            print("YOLO output:", result.stdout)
            
            # Find the label file
            label_path = os.path.join('yolo_output', predict_name, 'labels', f'temp_{unique_id}.txt')
            
            # Build tile vector from label file
            tile_vector = parse_yolo_txt(label_path)
            print(f"Tiles detected: {tile_vector}")
            
            if not tile_vector or tile_vector == ["No tiles detected"]:
                return jsonify({
                    "tiles": [],
                    "suggestion": "No mahjong tiles were detected in the image. Please ensure the image is clear and contains visible mahjong tiles."
                })
            
            # Create prompt for Gemini
            prompt = f"Given this Mahjong hand (Singapore mahjong rules): {', '.join(tile_vector)}, suggest the best tile to discard and explain why."
            
            # Get Gemini response
            print("Calling Gemini API...")
            gemini_response = call_gemini_api(prompt)
            
            return jsonify({
                "tiles": tile_vector,
                "suggestion": gemini_response,
                "status": "success"
            })
            
        except subprocess.CalledProcessError as e:
            print(f"YOLO error: {e}")
            print(f"YOLO stderr: {e.stderr}")
            return jsonify({"error": f"YOLO detection failed: {e.stderr}"}), 500
            
        except Exception as e:
            print(f"Processing error: {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    print(f"Cleaned up: {temp_filename}")
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def parse_yolo_txt(filepath):
    """Parse YOLO detection results from text file"""
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
                        print(f"Detected tile: {tile_name} (class_id: {class_id})")
                    else:
                        print(f"Invalid class_id: {class_id}")
                except ValueError as e:
                    print(f"Error parsing class_id: {e}")
                    
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return ["No tiles detected"]
    except Exception as e:
        print(f"Error parsing YOLO file: {e}")
        return ["No tiles detected"]
    
    return tiles if tiles else ["No tiles detected"]

def call_gemini_api(prompt):
    """Call Gemini API for mahjong analysis"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set. Please set your API key."
        
        # Use Gemini 1.5 Flash model
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
        
        print(f"Sending request to Gemini API...")
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=json_data,
            timeout=30
        )
        
        print(f"Gemini API response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Gemini API error: {response.text}")
            return f"Gemini API error: {response.status_code} - {response.text}"
        
        response_data = response.json()
        
        # Extract the generated text
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        return "Error: Unexpected response format from Gemini API"
        
    except requests.exceptions.Timeout:
        return "Error: Gemini API request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: Gemini API request failed - {str(e)}"
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error calling Gemini API: {str(e)}"

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('yolo_output', exist_ok=True)
    
    print("Starting Mahjong Analysis Server...")
    print("Make sure you have:")
    print("1. YOLO weights at: yolo_weights/best.pt")
    print("2. GEMINI_API_KEY environment variable set")
    print("3. Flask-CORS installed: pip install flask-cors")
    
    app.run(debug=True, host='0.0.0.0', port=5001)