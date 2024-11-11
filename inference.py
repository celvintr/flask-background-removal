from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from MODNet.src.models.modnet import MODNet
import cv2

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Crear carpetas de carga y procesadas si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Cargar el modelo MODNet pre-entrenado y configurar DataParallel
modnet = MODNet(backbone_pretrained=False)
modnet = torch.nn.DataParallel(modnet)  # Configuración de DataParallel
modnet.load_state_dict(torch.load('MODNet/modnet_photographic_portrait_matting.ckpt', map_location=torch.device('cpu')), strict=False)
modnet.eval()

def remove_background(input_path, output_path):
    ref_size = 512  # Tamaño de referencia para la red

    # Definir la transformación de imagen a tensor
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Cargar la imagen original
    image = Image.open(input_path).convert('RGB')
    original_size = image.size  # Obtener las dimensiones originales de la imagen

    # Convertir la imagen a un tensor
    im = im_transform(image).unsqueeze(0)

    # Redimensionar la imagen para la entrada del modelo MODNet
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

    # Inference
    with torch.no_grad():
        _, _, matte = modnet(im, True)

    # Redimensionar la máscara a las dimensiones originales de la imagen
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte = (matte * 255).astype(np.uint8)

    # Redimensionar la máscara a las dimensiones originales de la imagen
    matte = cv2.resize(matte, original_size, interpolation=cv2.INTER_LINEAR)
    image_np = np.array(image)

    # Crear una imagen RGBA
    rgba_image = np.dstack((image_np, matte))

    # Guardar la imagen resultante con fondo transparente
    output_image = Image.fromarray(rgba_image, 'RGBA')
    output_image.save(output_path, "PNG")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)

        if file:
            # Guardar la imagen en la carpeta de uploads
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Procesar la imagen utilizando MODNet
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{file.filename}")
            remove_background(file_path, processed_path)

            # Reemplazar las barras invertidas con barras normales para la URL
            processed_image_url = url_for('static', filename=f"processed/processed_{file.filename}")

            return render_template("index.html", processed_image=processed_image_url)
    
    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
