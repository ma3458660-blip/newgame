FROM python:3.11-slim

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nvidia-cuda-runtime-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cudnn-cu11==8.9.6.50

# Expose the CUDA 11.8 runtime libs to onnxruntime-gpu 1.18 (built against CUDA 11.8).
# If any lib fails to load, ORT falls back to CPUExecutionProvider, so this can only help.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# Face detector + embedder + landmark models (baked in so cold starts are fast)
RUN wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/bl.zip \
    && mkdir -p /root/.insightface/models/buffalo_l \
    && unzip -o /tmp/bl.zip -d /root/.insightface/models/ \
    && rm /tmp/bl.zip \
    && (test -f /root/.insightface/models/buffalo_l/det_10g.onnx \
        || for f in /root/.insightface/models/*.onnx; do mv "$f" /root/.insightface/models/buffalo_l/; done) \
    && ls -la /root/.insightface/models/buffalo_l/

# The swap model. insightface's get_model("inswapper_128.onnx") loads the FILE
# directly at ~/.insightface/models/inswapper_128.onnx. Download from the
# HuggingFace mirror (verified live, real 554MB file). The final size test makes
# sure the build fails loudly if the download came back wrong instead of
# shipping a worker that crashes on boot.
RUN mkdir -p /root/.insightface/models \
    && wget -q -O /root/.insightface/models/inswapper_128.onnx \
        https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
    && ls -la /root/.insightface/models/inswapper_128.onnx \
    && test "$(stat -c%s /root/.insightface/models/inswapper_128.onnx)" -gt 500000000

COPY handler.py /

CMD ["python", "-u", "/handler.py"]
