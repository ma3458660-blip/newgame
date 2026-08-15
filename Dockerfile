FROM runpod/base:0.4.7-cuda11.8.0

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

# Face detector + embedder + landmark models (baked in so cold starts are fast)
RUN wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/bl.zip \
    && mkdir -p /root/.insightface/models/buffalo_l \
    && unzip -o /tmp/bl.zip -d /root/.insightface/models/ \
    && rm /tmp/bl.zip \
    && (test -f /root/.insightface/models/buffalo_l/det_10g.onnx \
        || for f in /root/.insightface/models/*.onnx; do mv "$f" /root/.insightface/models/buffalo_l/; done) \
    && ls /root/.insightface/models/buffalo_l/

# The swap model. insightface expects it at ~/.insightface/models/inswapper_128.onnx/inswapper_128.onnx
RUN mkdir -p /root/.insightface/models/inswapper_128.onnx \
    && wget -q -O /root/.insightface/models/inswapper_128.onnx/inswapper_128.onnx \
        https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
    && ls -la /root/.insightface/models/inswapper_128.onnx/

COPY handler.py /

CMD ["python", "handler.py"]
