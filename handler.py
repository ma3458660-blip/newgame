import base64
import io
import urllib.request

import cv2
import numpy as np
import runpod
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# insightface's get_model treats a ".onnx" name as a raw path (relative to CWD),
# ignoring ~/.insightface/models/ entirely. So pass the FULL path to the model
# that the Dockerfile bakes in at /root/.insightface/models/inswapper_128.onnx.
swapper = get_model("/root/.insightface/models/inswapper_128.onnx", download=False, download_zip=False,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def decode_image(data):
    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        elif data.startswith("http"):
            req = urllib.request.Request(data, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not decode image from URL")
            return img
        raw = base64.b64decode(data)
    else:
        raw = bytes(data)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image data")
    return img


def best_face(img):
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score)


def handler(job):
    inp = job["input"]
    try:
        source = decode_image(inp.get("source"))
        target = decode_image(inp.get("target"))

        src_face = best_face(source)
        tgt_face = best_face(target)
        if src_face is None or tgt_face is None:
            return {"error": "no face detected in source or target image"}

        result = swapper.get(target, tgt_face, src_face, paste_back=True)

        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return {"error": "failed to encode result"}
        return {"image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
