import runpod
import base64
import os
import sys
from io import BytesIO
from PIL import Image

# 0. SUNTIKAN JALUR LINTAS BRUTAL (Cari folder trellis di seluruh lokasi umum RunPod)
sys.path.extend(["/app", "/workspace", "/workspace/TRELLIS", "/app/TRELLIS", os.path.dirname(__file__)])

pipeline = None
error_log = ""

# 1. INIT MODEL (Pakai pelindung try-except biar mesin nggak meledak kalau modul hilang)
try:
    print("Memuat model microsoft/TRELLIS.2-4B...")
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print("Model sukses dimuat!")
except Exception as e:
    error_log = str(e)
    print(f"CRASH SAAT LOADING: {error_log}")

def process_3d(job):
    # Kalau mesinnya gagal muat model di awal, lapor langsung ke Web (mesin tetep hidup)
    if pipeline is None:
        return {"error": f"Sistem gagal menemukan/memuat AI: {error_log}"}

    job_input = job['input']
    image_base64 = job_input.get('image_base64')
    
    if not image_base64:
        return {"error": "Masukkan image_base64"}

    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data))

        outputs = pipeline(img)

        glb_path = "/tmp/hasil_generate.glb"
        outputs.save_glb(glb_path)
        
        with open(glb_path, "rb") as f:
            glb_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        if os.path.exists(glb_path):
            os.remove(glb_path)

        return {
            "status": "success",
            "model_base64": glb_base64
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": process_3d})
