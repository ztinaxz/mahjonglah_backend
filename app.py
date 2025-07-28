from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import base64
import uuid

app = Flask(__name__)
CORS(app)

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

        try:
            # Read image data
            image_data = image.read()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Get image format
            image_format = image.content_type
            if not image_format:
                # Try to determine from filename
                filename_lower = image.filename.lower()
                if filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg'):
                    image_format = 'image/jpeg'
                elif filename_lower.endswith('.png'):
                    image_format = 'image/png'
                elif filename_lower.endswith('.webp'):
                    image_format = 'image/webp'
                else:
                    image_format = 'image/jpeg'  # default
            
            print(f"Image format: {image_format}")
            print("Sending image to Gemini...")

            # Call Gemini with image
            prompt = "Study the Chinese mahjong rules and suggest one tile to discard with a short explanation. Remember that 3 of the same tiles is good, 3 cconsecutive tiles is good too, do not discard them"
            gemini_response = call_gemini_with_image(prompt, image_base64, image_format)

            return jsonify({
                "suggestion": gemini_response,
                "status": "success"
            })

        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def call_gemini_with_image(prompt, image_base64, image_format):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        json_data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": image_format,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
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
            print(f"Gemini error response: {response.text}")
            return f"Gemini API error: {response.status_code} - {response.text}"

        response_data = response.json()
        print(f"Gemini response: {response_data}")
        
        if 'candidates' in response_data and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                return parts[0]['text'] if parts else "No response from Gemini."
            else:
                return "Gemini response format issue - no content found."

        return "Unexpected Gemini API response format."

    except requests.exceptions.Timeout:
        return "Gemini API request timed out."
    except Exception as e:
        print(f"Gemini exception: {e}")
        return f"Gemini API error: {str(e)}"

if __name__ == '__main__':
    print("MahjongLah backend running...")
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)