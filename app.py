"""
AI 图片超清增强 - 本地服务器
启动: python app.py
打开: http://localhost:5000
"""
import json, os, time, base64, io
from flask import Flask, request, jsonify, send_from_directory
import requests as req

app = Flask(__name__, static_folder='.')

REPLICATE_API = "https://api.replicate.com/v1"

MODELS = {
    "xinntao/realesrgan": {
        "name": "Real-ESRGAN",
        "desc": "通用超分，速度快，动漫/老照片强，免费可用",
        "tag": "快速通用",
        "version": "1b976a4d456ed9e4d1a846597b7614e79eadad3032e9124fa63859db0fd59b56",
        "input": {"scale": 4, "face_enhance": False, "version": "General - v3"},
        "input_key": "img",
    },
    "cjwbw/supir": {
        "name": "SUPIR",
        "desc": "顶级照片增强，细节最逼真（需付费）",
        "tag": "效果最佳",
        "version": "1302b550b4f7681da87ed0e405016d443fe1fafd64dabce6673401855a5039b5",
        "input": {"upscale": 2, "min_size": 2048, "use_llava": True, "color_fix_type": "Wavelet", "model_name": "SUPIR-v0F"},
        "input_key": "image",
    },
    "nightmareai/real-esrgan": {
        "name": "Real-ESRGAN v2",
        "desc": "增强版，人脸修复好（需付费）",
        "tag": "人脸增强",
        "version": "b3ef194191d13140337468c916c2c5b96dd0cb06dffc032a022a31807f6a5ea8",
        "input": {"scale": 4, "face_enhance": True},
        "input_key": "image",
    },
}


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/models')
def get_models():
    return jsonify(MODELS)


@app.route('/api/upscale', methods=['POST'])
def upscale():
    data = request.json
    api_key = data.get('api_key', '').strip()
    model_id = data.get('model', 'xinntao/realesrgan')
    image_b64 = data.get('image', '')  # base64 data URL

    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400
    if not image_b64:
        return jsonify({"error": "请提供图片"}), 400

    model = MODELS.get(model_id)
    if not model:
        return jsonify({"error": f"未知模型: {model_id}"}), 400

    headers = {"Authorization": f"Token {api_key}"}

    try:
        # Step 1: Decode base64 image and upload to Replicate
        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]

        image_bytes = base64.b64decode(image_b64)

        upload_resp = req.post(
            f"{REPLICATE_API}/files",
            headers=headers,
            files={"content": ("image.png", image_bytes, "image/png")},
        )
        upload_resp.raise_for_status()
        file_url = upload_resp.json()["urls"]["get"]

        # Step 2: Create prediction
        input_data = {**model["input"]}
        input_data[model["input_key"]] = file_url

        pred_resp = req.post(
            f"{REPLICATE_API}/predictions",
            headers={**headers, "Content-Type": "application/json"},
            json={"version": model["version"], "input": input_data},
        )
        pred_resp.raise_for_status()
        prediction = pred_resp.json()

        # Step 3: Return prediction ID for polling
        return jsonify({
            "prediction_id": prediction["id"],
            "status": prediction["status"],
        })

    except req.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = str(e)
        return jsonify({"error": detail or str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/poll/<prediction_id>')
def poll(prediction_id):
    api_key = request.headers.get("Authorization", "").replace("Token ", "")
    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400

    try:
        resp = req.get(
            f"{REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Token {api_key}"},
        )
        resp.raise_for_status()
        pred = resp.json()

        result = {
            "status": pred["status"],
            "prediction_id": pred["id"],
        }

        if pred["status"] == "succeeded":
            output = pred.get("output")
            if isinstance(output, list):
                output = output[0] if output else None
            result["output"] = output
        elif pred["status"] in ("failed", "canceled"):
            result["error"] = pred.get("error", f"任务 {pred['status']}")
        elif pred["status"] == "processing":
            result["logs"] = pred.get("logs", "")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/download')
def download():
    """Proxy download to avoid CORS issues with result URLs"""
    url = request.args.get("url", "")
    api_key = request.args.get("key", "")

    if not url:
        return jsonify({"error": "URL required"}), 400

    try:
        headers = {}
        if "api.replicate.com" in url and api_key:
            headers["Authorization"] = f"Token {api_key}"

        resp = req.get(url, headers=headers, stream=True)
        resp.raise_for_status()

        # Forward the content
        from flask import Response
        return Response(
            resp.content,
            content_type=resp.headers.get("content-type", "image/png"),
            headers={
                "Content-Disposition": "attachment; filename=upscaled.png",
                "Content-Length": str(len(resp.content)),
            },
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  AI 图片超清增强")
    print("  打开浏览器访问: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
