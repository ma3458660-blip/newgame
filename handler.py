import base64
import io
import sys
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

# --- Face restoration (GFPGAN), fully optional ---
# basicsr (gfpgan dep) imports the old module name torchvision.transforms.functional_tensor,
# which torchvision >=0.16 removed; alias it to the new functional module.
# If anything here fails, the worker still serves plain swaps (gfpganer stays None).
gfpganer = None
try:
    import torchvision
    import torchvision.transforms.functional as _tv_f
    sys.modules["torchvision.transforms.functional_tensor"] = _tv_f
    sys.modules["torchvision.transforms.functional_pil"] = _tv_f
    from gfpgan import GFPGANer
    gfpganer = GFPGANer(model_path="/root/gfpgan/GFPGANv1.4.pth", upscale=1,
                        arch="clean", channel_multiplier=2, bg_upsampler=None)
    print("GFPGAN loaded OK")
except Exception as e:
    print("GFPGAN unavailable, serving raw swaps:", e)


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
    # Largest face, not highest score: a bigger face produces a better ArcFace
    # embedding, which is what inswapper uses to transfer identity.
    faces = app.get(img)
    if not faces:
        return None

    def area(f):
        b = f.bbox.astype(int)
        return max(1, b[2] - b[0]) * max(1, b[3] - b[1])

    return max(faces, key=area)


def build_source_face(sources_raw):
    # Average the ArcFace embeddings of all reference images of the same person:
    # the averaged identity is usually more stable and closer to the reference
    # than any single photo.
    embs = []
    for s in sources_raw:
        try:
            f = best_face(decode_image(s))
            if f is not None:
                embs.append(f.normed_embedding)
        except Exception as e:
            print("source decode failed:", e)
    if not embs:
        return None
    avg = np.mean(embs, axis=0)
    avg = avg / np.linalg.norm(avg)

    class FakeFace:
        pass

    ff = FakeFace()
    ff.normed_embedding = avg
    return ff


def handler(job):
    inp = job["input"]
    try:
        target_raw = inp.get("target")
        sources_raw = inp.get("sources") or [inp.get("source")]

        target = decode_image(target_raw)
        tgt_face = best_face(target)
        if tgt_face is None:
            return {"error": "no face detected in target image"}

        src_face = build_source_face(sources_raw)
        if src_face is None:
            return {"error": "no face detected in any source image"}

        result = swapper.get(target, tgt_face, src_face, paste_back=True)

        try:
            if gfpganer is not None:
                _, _, result = gfpganer.enhance(result, has_aligned=False,
                                                only_center_face=False, paste_back=True)
        except Exception as e:
            print("GFPGAN enhance failed, using raw swap:", e)

        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return {"error": "failed to encode result"}
        return {"image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
