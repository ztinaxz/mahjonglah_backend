from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import requests
import uuid
import math

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

        try:
            # Save image
            image.save(temp_filename)
            print(f"Image saved to {temp_filename}")

            # Run YOLO detection with confidence threshold
            print("Running YOLO...")
            results = model(temp_filename, 
                            conf=0.4,      # Lower initial confidence for 14-tile constraint
                            iou=0.45,      # NMS IoU threshold  
                            max_det=20)    # Allow more detections initially, filter later
            print("YOLO done.")

            # Process YOLO results directly
            tile_vector = process_yolo_results(results[0])

            # Call Gemini with detected tiles (text only, not image)
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

def calculate_distance(box1, box2):
    """Calculate distance between two bounding box centers"""
    x1, y1 = box1[0], box1[1]  # Center coordinates
    x2, y2 = box2[0], box2[1]  # Center coordinates
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    # Convert from center format (x, y, w, h) to corner format (x1, y1, x2, y2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    box1_corners = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    box2_corners = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]
    
    # Calculate intersection
    x_left = max(box1_corners[0], box2_corners[0])
    y_top = max(box1_corners[1], box2_corners[1])
    x_right = min(box1_corners[2], box2_corners[2])
    y_bottom = min(box1_corners[3], box2_corners[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def process_yolo_results(result, target_tile_count=14):
    """Process YOLO results directly from the model output"""
    detections = []
    
    # Extract boxes, confidences, and class IDs from YOLO result
    if result.boxes is not None:
        boxes = result.boxes.xywh.cpu().numpy()  # Get boxes in xywh format
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if 0 <= class_id < len(tile_classes):
                detections.append({
                    'class_id': class_id,
                    'tile_name': tile_classes[class_id],
                    'bbox': box,  # [x_center, y_center, width, height]
                    'confidence': conf,
                    'used': False
                })
                print(f"Detected: {tile_classes[class_id]} (conf: {conf:.3f})")
    
    if not detections:
        print("No tiles detected by YOLO")
        return []
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Apply deduplication using same logic as before
    final_tiles = []
    
    for detection in detections:
        if detection['used']:
            continue
            
        # Check if this detection overlaps significantly with any already selected detection
        is_duplicate = False
        
        for other in detections:
            if other['used'] and other != detection:
                # Same class and high IoU = likely duplicate
                if (detection['class_id'] == other['class_id'] and 
                    calculate_iou(detection['bbox'], other['bbox']) > 0.3):
                    is_duplicate = True
                    print(f"Filtering duplicate {detection['tile_name']} due to IoU overlap")
                    break
                
                # Different class but very close proximity = possible misclassification
                elif calculate_distance(detection['bbox'], other['bbox']) < 0.05:
                    is_duplicate = True
                    print(f"Filtering {detection['tile_name']} due to proximity to {other['tile_name']}")
                    break
        
        if not is_duplicate:
            final_tiles.append(detection['tile_name'])
            detection['used'] = True
            print(f"Accepted: {detection['tile_name']}")
    
    print(f"Final tile count: {len(final_tiles)}")
    return final_tiles

def parse_yolo_txt_with_deduplication(filepath, confidence_threshold=0.4, iou_threshold=0.3, distance_threshold=0.05, target_tile_count=14):
    """Parse YOLO output with deduplication based on IoU and distance, optimized for 14-tile mahjong hands"""
    detections = []
    
    try:
        if not os.path.exists(filepath):
            print(f"Label file not found: {filepath}")
            return []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("Label file is empty")
            return []

        # Parse all detections first
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:  # class_id, x, y, w, h, confidence
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    confidence = float(parts[5])
                    
                    # Filter by confidence
                    if confidence >= confidence_threshold and 0 <= class_id < len(tile_classes):
                        detections.append({
                            'class_id': class_id,
                            'tile_name': tile_classes[class_id],
                            'bbox': [x, y, w, h],
                            'confidence': confidence,
                            'used': False
                        })
                        print(f"Detected: {tile_classes[class_id]} (conf: {confidence:.3f})")
                
                except (ValueError, IndexError) as e:
                    print(f"Parsing error for line '{line.strip()}': {e}")

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If we have way too many detections, be more strict with confidence
        if len(detections) > target_tile_count * 2:
            adjusted_threshold = min(0.7, max(confidence_threshold, 
                                           sorted([d['confidence'] for d in detections], reverse=True)[target_tile_count]))
            detections = [d for d in detections if d['confidence'] >= adjusted_threshold]
            print(f"Too many detections, adjusted confidence threshold to {adjusted_threshold:.3f}")
        
        # Sort by horizontal position (left to right) for better spatial understanding
        detections.sort(key=lambda x: x['bbox'][0])  # Sort by x-coordinate
        
        # Deduplicate using Non-Maximum Suppression approach
        final_tiles = []
        
        for detection in detections:
            if detection['used']:
                continue
                
            # Check if this detection overlaps significantly with any already selected detection
            is_duplicate = False
            
            for other in detections:
                if other['used'] and other != detection:
                    # Same class and high IoU = likely duplicate
                    if (detection['class_id'] == other['class_id'] and 
                        calculate_iou(detection['bbox'], other['bbox']) > iou_threshold):
                        is_duplicate = True
                        print(f"Filtering duplicate {detection['tile_name']} due to IoU overlap")
                        break
                    
                    # Different class but very close proximity = possible misclassification
                    elif calculate_distance(detection['bbox'], other['bbox']) < distance_threshold:
                        is_duplicate = True
                        print(f"Filtering {detection['tile_name']} due to proximity to {other['tile_name']}")
                        break
            
            if not is_duplicate:
                final_tiles.append(detection['tile_name'])
                detection['used'] = True
                print(f"Accepted: {detection['tile_name']}")
        
        # Validate tile count and provide feedback
        if len(final_tiles) != target_tile_count:
            print(f"Warning: Detected {len(final_tiles)} tiles, expected {target_tile_count}")
            
            # If we have too few tiles, try lowering confidence threshold
            if len(final_tiles) < target_tile_count and confidence_threshold > 0.3:
                print("Trying with lower confidence threshold...")
                return parse_yolo_txt_with_deduplication(filepath, 
                                                       confidence_threshold=0.3, 
                                                       iou_threshold=iou_threshold, 
                                                       distance_threshold=distance_threshold,
                                                       target_tile_count=target_tile_count)
            
            # If we still have too many, be more aggressive with deduplication
            elif len(final_tiles) > target_tile_count:
                print("Still too many tiles, applying stricter deduplication...")
                return parse_yolo_txt_with_deduplication(filepath, 
                                                       confidence_threshold=confidence_threshold + 0.1, 
                                                       iou_threshold=0.2, 
                                                       distance_threshold=0.03,
                                                       target_tile_count=target_tile_count)

    except Exception as e:
        print(f"Error parsing label file: {e}")
        return []

    return final_tiles if final_tiles else []

def parse_yolo_txt(filepath):
    """Original parsing function - kept as backup"""
    tiles = []
    try:
        if not os.path.exists(filepath):
            print(f"Label file not found: {filepath}")
            return []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("Label file is empty")
            return []

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
        return []

    return tiles if tiles else []

def call_gemini_api(prompt):
    """Call Gemini API with text prompt only (no image)"""
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