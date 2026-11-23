import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from hailo_sdk_client import ClientRunner, InferenceContext

BASE = Path(__file__).resolve().parent
PROBES, OUTPUTS = BASE / "probes", BASE / "outputs"
MODEL = "triangular_grouped32_embedding_v256_d64"
V, D, TOKENS, SEED = 256, 64, 4, 42
GROUPS = 32
PRECISION = "a16_w16"

def reference_table():
    return np.random.default_rng(SEED).normal(0, 0.1, (V, D)).astype(np.float32)

def make_onnx(path):
    e = reference_table()
    ids = np.arange(V, dtype=np.float32)
    # Conv1 produces [t-i, i-t] for each vocabulary entry i.
    w1 = np.tile([1., -1.], V).astype(np.float32).reshape(2*V, 1, 1, 1)
    b1 = np.column_stack((-ids, ids)).astype(np.float32).reshape(-1)
    # Conv2 combines paired positive differences: 1-ReLU(t-i)-ReLU(i-t).
    # Its large weight array is sparse: exactly two nonzero weights per output channel.
    per_group = V // GROUPS
    w2 = np.zeros((V, 2*per_group, 1, 1), dtype=np.float32)
    for j in range(V):
        c = 2*(j % per_group)
        w2[j, c, 0, 0] = -1
        w2[j, c+1, 0, 0] = -1
    b2 = np.ones(V, dtype=np.float32)
    # Conv3 selects E[i] via the triangular one-hot-like activations.
    w3 = e.T[:, :, None, None].copy()
    nodes = [
        h.make_node("Conv", ["x", "W1", "B1"], ["distance_pre"]),
        h.make_node("Relu", ["distance_pre"], ["distance"]),
        h.make_node("Conv", ["distance", "W2", "B2"], ["selector_pre"], group=GROUPS),
        h.make_node("Relu", ["selector_pre"], ["selector"]),
        h.make_node("Conv", ["selector", "W3"], ["y"]),
    ]
    weights = [nh.from_array(value, name) for name, value in
               [("W1", w1), ("B1", b1), ("W2", w2), ("B2", b2), ("W3", w3)]]
    graph = h.make_graph(nodes, MODEL,
                         [h.make_tensor_value_info("x", T.FLOAT, [1, 1, 1, TOKENS])],
                         [h.make_tensor_value_info("y", T.FLOAT, [1, D, 1, TOKENS])],
                         initializer=weights)
    model = h.make_model(graph, producer_name="onnx2hailo", opset_imports=[h.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"[GENERATE] {path} ({path.stat().st_size:,} bytes)", flush=True)

def validate_onnx(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    e = reference_table()
    errors = []
    for start in range(0, V, TOKENS):
        ids = np.arange(start, start+TOKENS, dtype=np.float32).reshape(1, 1, 1, TOKENS)
        output = session.run(None, {"x": ids})[0]
        expected = e[start:start+TOKENS].T.reshape(1, D, 1, TOKENS)
        errors.append(float(np.max(np.abs(output-expected))))
    max_error = max(errors)
    print(f"[VALIDATE] All {V} tokens: FP32 maximum absolute error={max_error:.9g}", flush=True)
    if not np.isfinite(max_error) or max_error >= 1e-6:
        raise RuntimeError(f"ONNX localized lookup error {max_error} exceeds 1e-6")

def parse_onnx(onnx_path, har_path):
    runner = ClientRunner(hw_arch="hailo10h")
    runner.translate_onnx_model(str(onnx_path), MODEL)
    runner.save_har(str(har_path))
    print(f"[PARSE] {har_path}", flush=True)

def optimize_har(input_har, optimized_har):
    runner = ClientRunner(har=str(input_har))
    runner.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n"
                             "quantization_param({conv*}, precision_mode=a16_w16)")
    # Exactly 64 samples x 4 tokens; every ID 0..255 appears exactly once.
    # calib = np.random.default_rng(SEED).permutation(V).astype(np.float32).reshape(V//TOKENS, 1, TOKENS, 1)
    rng = np.random.default_rng(SEED)
    ids = np.arange(V, dtype=np.float32)
    rng.shuffle(ids)
    calib = ids.reshape(V // TOKENS, 1, TOKENS, 1)
    print(f"[OPTIMIZE] Calibration={calib.shape}, unique IDs={len(np.unique(calib))}, precision={PRECISION}", flush=True)
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
    raise ValueError(f"Unexpected emulator shape: {y.shape}")

def evaluate(optimized_har, metrics_path):
    runner = ClientRunner(har=str(optimized_har))
    rows = []
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for start in range(0, V, TOKENS):
            x = np.arange(start, start+TOKENS, dtype=np.float32).reshape(1, 1, TOKENS, 1)
            output = runner.infer(ctx, x)
            if isinstance(output, dict): output = next(iter(output.values()))
            if isinstance(output, (list, tuple)): output = output[0]
            rows.append(to_token_rows(output))
    actual, expected = np.concatenate(rows), reference_table()
    absolute = np.abs(actual-expected)
    reference_unit = expected / np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
    actual_unit = actual / np.maximum(np.linalg.norm(actual, axis=1, keepdims=True), 1e-12)
    cosine = np.sum(reference_unit * actual_unit, axis=1)
    nearest = np.argmax(actual_unit @ reference_unit.T, axis=1)
    mismatches = [{"token": int(i), "predicted": int(nearest[i])}
                  for i in np.flatnonzero(nearest != np.arange(V))]
    per_token_mae = absolute.mean(axis=1)
    metrics = {
        "model": MODEL, "precision": PRECISION, "tokens": V, "dimensions": D,
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "mean_cosine_similarity": float(cosine.mean()),
        "worst_cosine_similarity": float(cosine.min()),
        "nearest_cosine_correct": V-len(mismatches), "nearest_cosine_total": V,
        "mismatches": mismatches,
        "mean_absolute_error_by_32_token_block": [
            {"start": int(s), "end": int(min(s+32, V)-1),
             "mean_absolute_error": float(per_token_mae[s:s+32].mean())}
            for s in range(0, V, 32)
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2)+"\n")
    print("[EMULATOR] "+json.dumps(metrics, indent=2), flush=True)
    print(f"[EMULATOR] {metrics_path}", flush=True)

def main():
    PROBES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    onnx_path = PROBES / f"{MODEL}.onnx"
    raw_har = OUTPUTS / f"{MODEL}.har"
    optimized_har = OUTPUTS / f"{MODEL}_{PRECISION}_optimized.har"
    hef_path = OUTPUTS / f"{MODEL}_{PRECISION}.hef"
    metrics_path = OUTPUTS / f"{MODEL}_{PRECISION}_metrics.json"
    print("\n[1/6] Generate triangular ONNX", flush=True)
    make_onnx(onnx_path)
    print("\n[2/6] Validate ONNX against all 256 embeddings", flush=True)
    validate_onnx(onnx_path)
    print("\n[3/6] Parse ONNX to HAR", flush=True)
    parse_onnx(onnx_path, raw_har)
    print("\n[4/6] Optimize and quantize (A16W16)", flush=True)
    optimize_har(raw_har, optimized_har)
    print("\n[5/6] Compile HEF", flush=True)
    compile_har(optimized_har, hef_path)
    print("\n[6/6] Evaluate quantized SDK emulator", flush=True)
    evaluate(optimized_har, metrics_path)
    print("\n[SUCCESS] Triangular lookup pipeline completed", flush=True)

if __name__ == "__main__":
    main()
