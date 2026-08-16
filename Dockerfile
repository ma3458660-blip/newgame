FROM python:3.11-slim

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /

RUN wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/bl.zip \
    && mkdir -p /root/.insightface/models/buffalo_l \
    && unzip -o /tmp/bl.zip -d /root/.insightface/models/ \
    && rm /tmp/bl.zip \
    && (test -f /root/.insightface/models/buffalo_l/det_10g.onnx \
        || for f in /root/.insightface/models/*.onnx; do mv "$f" /root/.insightface/models/buffalo_l/; done) \
    && ls -la /root/.insightface/models/buffalo_l/

RUN mkdir -p /root/.insightface/models \
    && wget -q -O /root/.insightface/models/inswapper_128.onnx \
        https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
    && ls -la /root/.insightface/models/inswapper_128.onnx \
    && test "$(stat -c%s /root/.insightface/models/inswapper_128.onnx)" -gt 500000000

RUN mkdir -p /root/gfpgan /facexlib/weights \
    && wget -q -O /root/gfpgan/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
    && wget -q -O /facexlib/weights/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
    && wget -q -O /facexlib/weights/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth \
    && ls -la /root/gfpgan/ /facexlib/weights/ \
    && test "$(stat -c%s /root/gfpgan/GFPGANv1.4.pth)" -gt 300000000 \
    && test "$(stat -c%s /facexlib/weights/detection_Resnet50_Final.pth)" -gt 90000000 \
    && test "$(stat -c%s /facexlib/weights/parsing_parsenet.pth)" -gt 70000000

# torch = CPU build (GFPGAN only); numpy pinned in the gfpgan call so pip never
# installs numpy 2.x (crashes onnxruntime 1.18) -> no extra re-run step needed.
# scipy pinned to a numpy-1.24-compatible version -> no pip version backtracking.
# cu11 libs installed unpinned so no version typos can fail the build.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir gfpgan==1.3.8 numpy==1.24.4 \
    && pip install --no-cache-dir nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cudnn-cu11 nvidia-curand-cu11 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib

COPY handler.py /

CMD ["python", "-u", "/handler.py"]
