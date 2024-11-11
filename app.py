from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file

import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from MODNet.src.models.modnet import MODNet
import cv2
import rembg

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Crear carpetas de carga y procesadas si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Cargar el modelo MODNet pre-entrenado y configurar DataParallel
modnet = MODNet(backbone_pretrained=False)
modnet = torch.nn.DataParallel(modnet)
modnet.load_state_dict(torch.load('MODNet/modnet_photographic_portrait_matting.ckpt', map_location=torch.device('cpu')), strict=False)
modnet.eval()

def remove_background_person(input_path, output_path):
    ref_size = 512
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(input_path).convert('RGB')
    original_size = image.size

    im = im_transform(image).unsqueeze(0)

    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    with torch.no_grad():
        _, _, matte = modnet(im, True)

    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte = (matte * 255).astype(np.uint8)

    matte = cv2.resize(matte, original_size, interpolation=cv2.INTER_CUBIC)
    matte = cv2.GaussianBlur(matte, (5, 5), 0)

    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = image_np[:, :, None]
    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, 0:3]

    matte = np.repeat(matte[:, :, None], 3, axis=2) / 255
    foreground = image_np * matte + np.full(image_np.shape, 255) * (1 - matte)

    rgba_image = np.dstack((foreground, matte[..., 0] * 255))

    foreground_image = Image.fromarray(np.uint8(rgba_image), 'RGBA')
    foreground_image.save(output_path, "PNG")

def remove_background_object(input_path, output_path):
    input_image = Image.open(input_path)
    input_array = np.array(input_image)
    output_array = rembg.remove(input_array)
    output_image = Image.fromarray(output_array)

    # Redimensionar si el ancho es mayor a 800 px
    max_width = 800
    original_size = output_image.size
    if original_size[0] > max_width:
        ratio = max_width / float(original_size[0])
        new_height = int((float(original_size[1]) * float(ratio)))
        # Utiliza Image.Resampling.LANCZOS en lugar de Image.ANTIALIAS
        output_image = output_image.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Guardar la imagen optimizada con compresión iterativa para asegurar un tamaño menor a 1 MB
    temp_quality = 95
    while True:
        output_image.save(output_path, "PNG", optimize=True, quality=temp_quality)
        if os.path.getsize(output_path) <= 1 * 1024 * 1024 or temp_quality <= 10:
            break
        temp_quality -= 5



@app.route("/", methods=["GET", "POST"])
def index():
    if 'processed_images' not in session:
        session['processed_images'] = []

    if request.method == "POST":
        if 'file' not in request.files or 'image_type' not in request.form:
            return jsonify({"error": "No se seleccionó ningún archivo o tipo de imagen"}), 400

        file = request.files['file']
        image_type = request.form['image_type']

        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            output_filename = f"processed_{os.path.splitext(file.filename)[0]}.png"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

            if image_type == 'person':
                remove_background_person(file_path, processed_path)
            else:
                remove_background_object(file_path, processed_path)

            # Añadir la imagen procesada a la sesión
            session['processed_images'].append(output_filename)
            session.modified = True

            return jsonify({"filename": output_filename})

    return render_template("index.html", processed_images=session['processed_images'])



@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
