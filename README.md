# Face-swap RunPod worker

InsightFace face-swap (swap a real face photo's identity onto any generated image)
as a RunPod serverless endpoint. Pay per GPU-second only while a request runs.

## Deploy steps

1. Push these 4 files to a GitHub repo (or any git host RunPod supports).

2. RunPod console -> Serverless -> **New Endpoint**:
   - Name: e.g. `face-swap`
   - **GPU**: a cheap one — `RTX 4090` (fine) or `A4500`/`V100 16GB` (cheaper). Any of them is overkill but this is the cheapest class. Avoid the $1+/hr flagship cards.
   - **Container**: choose "Build from git repo", paste your repo URL + branch (build takes ~5-10 min the first time; later pushes rebuild incrementally).
   - **Workers**: Min 0 (scale-to-zero = $0.00 when idle), Max 1-2.
   - Keep worker idle timeout small (e.g. 5s) so workers die fast when idle.
   - Create.

3. Grab from the endpoint page: the **Endpoint ID** (`xxx...`) and from
   Settings -> API Keys your **API key** (starts `rp_...`).

4. Test it (substitute your own values; see "Making test images" below):

```bash
ENDPOINT_ID=YOUR_ENDPOINT_ID
KEY=YOUR_API_KEY

curl -X POST "https://api.runpod.io/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"source":"data:image/jpeg;base64,<SOURCE_FACE_BASE64>","target":"data:image/jpeg;base64,<TARGET_IMAGE_BASE64>"}}'
```

`source` = the real face whose identity you want (a clear frontal photo).
`target` = the image to swap that face onto. Both can also be plain `http(s)://`
URLs instead of base64.

Response: `{"output":{"image":"data:image/jpeg;base64,..."}}` or
`{"output":{"error":"..."}}` if no face was found in either image.

## Making test images

Easiest: two face photos on your phone/laptop, convert to base64:

```bash
base64 -w0 face.jpg   # macOS/Linux — paste output into the JSON above
```

The curl above is fine for a 1-2s swap (runsync). If the job needs longer than
the HTTP timeout, add `"timeout": 120` to the JSON body.

## Cost / latency notes

- Billed per GPU-second only while a request is actually running. A single swap
  is roughly $0.0002-0.0005. $4 ≈ hundreds of swaps.
- The first request after idle pays a **cold start**: container boots (~10-30s of
  GPU time) + loads models (~5s). Subsequent requests hit a warm worker instantly.
  For snappy chat-integrated use, keep workers warm with `Min Workers: 1` when
  you're actively using it, or accept the slow first swap after an idle gap.
- The 554MB inswapper model is baked into the image at build time so cold starts
  don't re-download it.

## Security

The API key is checked by the endpoint, so any client with the URL + key can call
it and spend your money. That's acceptable while the generator stays **private**.
If it ever goes public, proxy it through a Perchance server-plugin or issue
per-user keys with quotas.

## Troubleshooting

- **Build fails on pip**: Python version mismatch on the base image — try
  `runpod/base:0.4.5-cuda11.8.0` or `runpod/base:0.5.6-cuda12.1.0` with
  `onnxruntime-gpu==1.17.1`.
- **"no face detected"**: use a well-lit frontal face photo for `source`; for the
  target make sure the generated image has a visible face.
- **Slow first request**: that's the cold start; it's normal.
