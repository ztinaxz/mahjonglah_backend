from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import uuid
import base64
from PIL import Image
import io

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
            # Read and process the image
            image_data = image.read()
            
            # Optional: Resize image if too large (Gemini has size limits)
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Resize if image is too large (max 20MB for Gemini)
            max_size = (1024, 1024)  # Adjust as needed
            if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=85)
                image_data = buffer.getvalue()

            # Convert to base64 for Gemini API
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Call Gemini Vision API
            prompt = """Analyze this image of Mahjong tiles and provide:
1. List all the tiles you can see in the image (use standard mahjong notation like "1-bamboo", "2-dots", "east-wind", "red-dragon", etc.)
2. Based on Singapore Mahjong rules, suggest which tile to discard and explain your reasoning
3. If you cannot clearly identify the tiles, please say so

Please be specific about what tiles you see and provide strategic advice."""

            gemini_response = call_gemini_vision_api(prompt, base64_image)

            return jsonify({
                "suggestion": gemini_response,
                "status": "success",
                "analysis_method": "gemini_vision"
            })

        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def call_gemini_vision_api(prompt, base64_image):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        json_data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=json_data,
            timeout=30
        )

        print(f"Gemini Vision response code: {response.status_code}")
        if response.status_code != 200:
            print(f"Gemini API error response: {response.text}")
            return f"Gemini API error: {response.status_code} - {response.text}"

        response_data = response.json()
        if 'candidates' in response_data and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                return parts[0]['text'] if parts else "No response from Gemini."
            else:
                print(f"Unexpected response structure: {response_data}")
                return "Gemini API returned unexpected format."

        return "No candidates in Gemini response."

    except requests.exceptions.Timeout:
        return "Gemini API request timed out."
    except Exception as e:
        print(f"Gemini Vision exception: {e}")
        return f"Gemini Vision API error: {str(e)}"

# Keep the original text-based Gemini function as backup
def call_gemini_text_api(prompt):
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

        print(f"Gemini text response code: {response.status_code}")
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
        print(f"Gemini text exception: {e}")
        return f"Gemini text API error: {str(e)}"

if __name__ == '__main__':
    print("MahjongLah backend with Gemini Vision running...")
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)