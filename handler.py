import runpod
import requests
import base64
import os
import sys
from io import BytesIO
from PIL import Image

# 0. TAMBAHAN GPS: Paksa Python baca folder /app biar modul 'trellis' terbaca
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("/app")

# 1. INIT MODEL
print("Memuat model microsoft/TRELLIS.2-4B...")
from trellis.pipelines import TrellisImageTo3DPipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# ... (bagian def process_3d(job): ke bawah biarin aja sama kayak tadi) ...

def process_3d(job):
    job_input = job['input']
    image_base64 = job_input.get('image_base64')
    
    if not image_base64:
        return {"error": "Masukkan image_base64"}

    try:
        # 1. Hapus awalan data:image jika ada
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        # 2. Decode gambar dari web
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data))

        # 3. PROSES KE AI TRELLIS
        outputs = pipeline(img)

        # 4. Simpan hasil 3D
        glb_path = "/tmp/hasil_generate.glb"
        outputs.save_glb(glb_path)
        
        # 5. Baca file 3D dan jadikan Base64 untuk dikirim ke Web
        with open(glb_path, "rb") as f:
            glb_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        # 6. Bersihkan file temporary biar memori gak penuh
        if os.path.exists(glb_path):
            os.remove(glb_path)

        return {
            "status": "success",
            "model_base64": glb_base64
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": process_3d})
