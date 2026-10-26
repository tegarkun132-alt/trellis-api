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

# ... (bagian atas tetap sama) ...

def process_3d(job):
    job_input = job['input']
    image_base64 = job_input.get('image_base64')
    
    if not image_base64:
        return {"error": "Masukkan image_base64"}

    try:
        # Hapus awalan "data:image/png;base64," jika ada dari HTML
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        # Decode gambar dari Base64
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data))

        # ... (sisa kode proses Trellis ke bawah tetap sama) ...

        # 3. Proses Trellis (Uncomment kalau library Trellis udah di-install lengkap)
        # outputs = pipeline(img)
        
        glb_path = "/tmp/hasil_generate.glb"
        outputs.save_glb(glb_path)
        
        # DUMMY FILE UNTUK TESTING (Hapus 2 baris ini kalau model asli udah jalan)
        #with open(glb_path, "w") as f: f.write("dummy 3d data")

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
