from flask import (
    Flask, request, Response, send_file
)
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import torch


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_format = {"jpg", "jpeg", "png"}

def generate(file, pretrained, format):
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained)
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=512)

    binary = file.read()

    im_in = Image.open(BytesIO(binary)).convert("RGB")
    im_out = face2paint(model, im_in, side_by_side=False)
    buffer_out = BytesIO()
    im_out.save(buffer_out, format=format)
    buffer_out.seek(0)
    
    return buffer_out

@app.route('/animeganv2', methods=['POST'])
def animeganv2():
    try:
        file = request.files['file']
        pretrained = request.form['pretrained']
    except:
        return Response("Empty Field", status=400)
    
    filename = secure_filename(file.filename)
    ext = filename.split('.')[-1]

    if ext not in image_format:
        return Response("File Error", status=400)

    result = generate(file, pretrained, ext)
    return send_file(result, mimetype='image/' + ext)

@app.route('/health', methods=['GET'])
def main():
    return "ok"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")