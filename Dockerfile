FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install library
COPY requirements.txt .
RUN pip install -r requirements.txt huggingface_hub

# Bakar model Trellis langsung ke dalam Docker Image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/TRELLIS.2-4B')"

# Copy file handler
COPY handler.py .

CMD [ "python", "-u", "/app/handler.py" ]