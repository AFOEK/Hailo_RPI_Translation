import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from hailo_sdk_client import ClientRunner, InferenceContext

BASE = Path(__file__).resolve().parent
PROBES, OUTPUTS = BASE / "probes", BASE / "outputs"
MODEL = "spline_embedding"
V, D, TOKENS, SEED = 16, 8, 4, 42

def reference_table():
    return np.random.default_rng(SEED).normal(0, 0.1, (V, D)).astype(np.float32)

def make_onnx(path):
    e = reference_table()
    coeff = np.empty((V - 1, D), dtype=np.float32)
    coeff[0] = e[1] - e[0]
    coeff[1:] = e[2:] - 2 * e[1:-1] + e[:-2]
    w1 = np.ones((V - 1, 1, 1, 1), dtype=np.float32)
    b1 = -np.arange(V - 1, dtype=np.float32)
    w2 = coeff.T[:, :, None, None].copy()
    b2 = e[0].copy()
    nodes = [h.make_node("Conv", ["x", "W1", "B1"], ["a"]),
             h.make_node("Relu", ["a"], ["b"]),
             h.make_node("Conv", ["b", "W2", "B2"], ["y"])]
    weights = [nh.from_array(arr, name) for arr, name in
               [(w1, "W1"), (b1, "B1"), (w2, "W2"), (b2, "B2")]]
    graph = h.make_graph(nodes, MODEL,
                         [h.make_tensor_value_info("x", T.FLOAT, [1, 1, 1, TOKENS])],
                         [h.make_tensor_value_info("y", T.FLOAT, [1, D, 1, TOKENS])],
                         initializer=weights)
    model = h.make_model(graph, producer_name="onnx2hailo", opset_imports=[h.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[GENERATE] {path}", flush=True)

def validate_onnx(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    e = reference_table()
    errors = []
    for start in range(0, V, TOKENS):
        ids = np.arange(start, start + TOKENS, dtype=np.float32).reshape(1, 1, 1, TOKENS)
        output = session.run(None, {"x": ids})[0]
        expected = e[start:start + TOKENS].T.reshape(1, D, 1, TOKENS)
        errors.append(np.max(np.abs(output - expected)))
    max_error = float(max(errors))
    print(f"[VALIDATE] All {V} token IDs: max absolute error={max_error:.9g}", flush=True)
    if max_error >= 1e-5:
        raise RuntimeError("ONNX spline lookup does not reproduce the reference table")
    return max_error

def parse_onnx(onnx_path, har_path):
    runner = ClientRunner(hw_arch="hailo10h")
    runner.translate_onnx_model(str(onnx_path), MODEL)
    runner.save_har(str(har_path))
    print(f"[PARSE] {har_path}", flush=True)

def optimize_har(input_har, optimized_har, precision):
    runner = ClientRunner(har=str(input_har))
    script = "model_optimization_flavor(optimization_level=0, compression_level=0)"
    if precision == "a16_w16":
        script += "\nquantization_param({conv*}, precision_mode=a16_w16)"
    runner.load_model_script(script)
    calib = np.random.default_rng(SEED).integers(0, V, size=(256, 1, TOKENS, 1)).astype(np.float32)
    print(f"[OPTIMIZE] Calibration shape={calib.shape}, precision={precision}", flush=True)
    runner.optimize(calib)
    runner.save_har(str(optimized_har))
    print(f"[OPTIMIZE] {optimized_har}", flush=True)

def compile_har(optimized_har, hef_path):
    runner = ClientRunner(har=str(optimized_har))
    hef_path.write_bytes(runner.compile())
    print(f"[COMPILE] {hef_path} ({hef_path.stat().st_size:,} bytes)", flush=True)

def to_token_rows(output):
    y = np.asarray(output)
    if y.shape == (1, 1, TOKENS, D):
        return y.reshape(TOKENS, D)
    if y.shape == (1, D, 1, TOKENS):
        return y.transpose(0, 2, 3, 1).reshape(TOKENS, D)
    raise ValueError(f"Unexpected emulator output shape {y.shape}")

def evaluate(optimized_har, metrics_path, precision):
    runner = ClientRunner(har=str(optimized_har))
    rows = []
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for start in range(0, V, TOKENS):
            inputs = np.arange(start, start + TOKENS, dtype=np.float32).reshape(1, 1, TOKENS, 1)
            output = runner.infer(ctx, inputs)
            if isinstance(output, dict):
                output = next(iter(output.values()))
            if isinstance(output, (tuple, list)):
                output = output[0]
            rows.append(to_token_rows(output))
    actual = np.concatenate(rows, axis=0)
    expected = reference_table()
    absolute = np.abs(actual - expected)
    ref_norm = expected / np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
    out_norm = actual / np.maximum(np.linalg.norm(actual, axis=1, keepdims=True), 1e-12)
    cosine = np.sum(ref_norm * out_norm, axis=1)
    nearest = np.argmax(out_norm @ ref_norm.T, axis=1)
    metrics = {"precision": precision, "tokens": V, "dimensions": D,
               "maximum_absolute_error": float(absolute.max()),
               "mean_absolute_error": float(absolute.mean()),
               "mean_cosine_similarity": float(cosine.mean()),
               "worst_cosine_similarity": float(cosine.min()),
               "nearest_cosine_correct": int(np.sum(nearest == np.arange(V))),
               "nearest_cosine_total": V,
               "predicted_token_ids": nearest.tolist()}
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print("[EMULATOR] " + json.dumps(metrics, indent=2), flush=True)
    print(f"[EMULATOR] Saved metrics: {metrics_path}", flush=True)


def main():
    precision = "a16_w16"  #a16_w16 | int8
    PROBES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    onnx_path = PROBES / f"{MODEL}.onnx"
    raw_har = OUTPUTS / f"{MODEL}.har"
    optimized_har = OUTPUTS / f"{MODEL}_{precision}_optimized.har"
    hef_path = OUTPUTS / f"{MODEL}_{precision}.hef"
    metrics_path = OUTPUTS / f"{MODEL}_{precision}_metrics.json"

    print("\n[1/6] Generating ONNX model", flush=True)
    make_onnx(onnx_path)
    print("\n[2/6] Validating ONNX model", flush=True)
    validate_onnx(onnx_path)
    print("\n[3/6] Parsing ONNX to HAR", flush=True)
    parse_onnx(onnx_path, raw_har)
    print("\n[4/6] Optimizing and quantizing", flush=True)
    optimize_har(raw_har, optimized_har, precision)
    print("\n[5/6] Compiling HEF", flush=True)
    compile_har(optimized_har, hef_path)
    print("\n[6/6] Evaluating quantized model", flush=True)
    evaluate(optimized_har, metrics_path, precision)
    print("\n[SUCCESS] Complete ONNX-to-Hailo pipeline finished", flush=True)

if __name__ == "__main__":
    main()