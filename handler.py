import runpod
import requests
import base64
import os
from io import BytesIO
from PIL import Image

# 1. INIT MODEL (Berjalan sekali saat mesin nyala biar proses cepat)
print("Memuat model microsoft/TRELLIS.2-4B...")
# Catatan: Sesuaikan import ini dengan dokumentasi resmi repo GitHub Trellis
# from trellis.pipelines import TrellisImageTo3DPipeline
# pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
# pipeline.to("cuda")

def process_3d(job):
    job_input = job['input']
    image_url = job_input.get('image_url')
    
    if not image_url:
        return {"error": "Masukkan image_url"}

    try:
        # 2. Download Gambar dari Web Lu
        req = requests.get(image_url)
        img = Image.open(BytesIO(req.content))

        # 3. Proses Trellis (Uncomment kalau library Trellis udah di-install lengkap)
        # outputs = pipeline(img)
        
        glb_path = "/tmp/hasil_generate.glb"
        # outputs.save_glb(glb_path)
        
        # DUMMY FILE UNTUK TESTING (Hapus 2 baris ini kalau model asli udah jalan)
        with open(glb_path, "w") as f: f.write("dummy 3d data")

        # 4. Ubah file .glb ke Base64 biar bisa langsung dikirim ke Web/Frontend
        with open(glb_path, "rb") as f:
            glb_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        # Bersihkan file temporary biar memori gak penuh
        os.remove(glb_path)

        return {
            "status": "success",
            "model_base64": glb_base64
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": process_3d})