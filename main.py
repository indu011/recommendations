import os
import tempfile
from google import generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import time
load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv('API_KEY')
genai.configure(api_key=api_key)  

labels = [
    "formal", "semi-formal", "party", "cocktail", "evening", "work/office", 
    "wedding guest", "beach/resort", "mini", "knee-length", "midi", "tea-length", "maxi", 
    "floor-length", "v-neck", "round neck", "scoop neck", "square neck", "sweetheart", 
    "halter neck", "strapless", "off-the-shoulder", "one-shoulder", "boat neck", "cowl neck", 
    "high neck", "sleeveless", "spaghetti strap", "cap sleeve", "short sleeve", "3/4 sleeve", 
    "long sleeve", "puff sleeve", "bell sleeve", "bishop sleeve", "flutter sleeve", "cold shoulder", 
    "solid", "floral", "striped", "polka dot", "plaid", "animal print", "geometric", "paisley", 
    "abstract", "embroidered", "loose", "relaxed", "flowy", "fitted", "bodycon", "a-line", "straight",
    "sweater", "skirt", "shirt dress", "tunic dress", "top", "pants", "bridal", "saree", "tie neck", 
    "summer dress", "checked", "eyelet", "elastic waist", "fit-and-flare", "boho", "kurta", "printed"
]

@app.route('/api/health')
def health_check():
    try:
        return jsonify({"status": "healthy", "message": "API is running smoothly!"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "message": str(e)}), 500

import time

@app.route('/api/classify_dress', methods=['POST'])
def classify_dress():
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({"error": "No image provided"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_file:
            image_file.save(temp_file.name)
            temp_file_path = temp_file.name

        uploaded_file = genai.upload_file(temp_file_path)

        labels_str = ", ".join(labels)
        prompt = f'''
        Recognize the image and classify it based on the following labels:
        {labels_str}
        
        Only return:
        1. Whether it's a clothing item ("yes" or "no") in the first line in lowercase
        2. If it's a clothing item, in the next line return an array with the appropriate labels it matches (from the list provided above)
        If it's not a clothing item, just return "no" and no second line, only if it's a yes, we get a second line

        Further instructions:
        - Only return major labels, you don't have to cross check each label
        - Label count should be as low as possible, minimum is 2, maximum is 3
        '''

        model = genai.GenerativeModel('gemini-1.5-flash')
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content([uploaded_file, prompt])
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(2) 
                    continue
                else:
                    raise e

        os.unlink(temp_file_path)

        return jsonify({"response": response.text.strip()})

    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
