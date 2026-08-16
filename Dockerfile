FROM python:3.11-slim

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /

# Face detector + embedder + landmark models (baked in so cold starts are fast)
RUN wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/bl.zip \
    && mkdir -p /root/.insightface/models/buffalo_l \
    && unzip -o /tmp/bl.zip -d /root/.insightface/models/ \
    && rm /tmp/bl.zip \
    && (test -f /root/.insightface/models/buffalo_l/det_10g.onnx \
        || for f in /root/.insightface/models/*.onnx; do mv "$f" /root/.insightface/models/buffalo_l/; done) \
    && ls -la /root/.insightface/models/buffalo_l/

# The swap model. insightface's get_model treats a ".onnx" name as a raw path
# (relative to CWD), so we bake the file here and pass its FULL path from handler.py.
RUN mkdir -p /root/.insightface/models \
    && wget -q -O /root/.insightface/models/inswapper_128.onnx \
        https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
    && ls -la /root/.insightface/models/inswapper_128.onnx \
    && test "$(stat -c%s /root/.insightface/models/inswapper_128.onnx)" -gt 500000000

# GFPGAN face-restoration model + the facexlib weights it auto-loads.
# Paths match what gfpgan/facexlib expect when running with CWD=/. Size checks
# fail the build loudly if any download came back wrong.
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

# Python deps. torch is the CPU-only build (small ~190MB). gfpgan pulls
# basicsr + facexlib AND upgrades numpy to 2.x, which crashes onnxruntime 1.18
# (compiled against numpy 1.x) — so we re-run requirements.txt LAST to force all
# pinned versions (numpy==1.24.4 etc.) back into place.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir gfpgan==1.3.8 \
    && pip install --no-cache-dir -r requirements.txt

COPY handler.py /

CMD ["python", "-u", "/handler.py"]
