from flask import (
    Flask, request, Response, send_file, render_template
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
image_format = {"image/jpg", "image/jpeg", "image/png"}

model_list = {"celeba_distill", "face_paint_512_v1", "face_paint_512_v2", "paprika"}

models = {
    "celeba_distill": torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill", device=device),
    "face_paint_512_v1": torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1", device=device),
    "face_paint_512_v2": torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", device=device),
    "paprika": torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika", device=device),
}
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=512)

def generate(file, pretrained, format):
    binary = file.read()
    im_in = Image.open(BytesIO(binary)).convert("RGB")
    im_out = face2paint(models[pretrained], im_in, side_by_side=False)
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
    
    if pretrained not in model_list:
        return Response("Does not exist model", status=400)

    if file.content_type not in image_format:
        return Response("File Error", status=400)

    result = generate(file, pretrained, file.content_type.split('/')[-1])
    return send_file(result, mimetype=file.content_type)

@app.route('/health', methods=['GET'])
def health_check():
    return "ok"

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")