from pathlib import Path
from hailo_sdk_client import ClientRunner
import json

results = {}

for path in sorted(Path("probes").glob("*.onnx")):
    name = path.stem
    print(f"\n=== Testing {name} ===", flush=True)
    try:
        runner = ClientRunner(hw_arch="hailo10h")
        runner.translate_onnx_model(str(path), name)
        runner.save_har(f"outputs/{name}.har")
        results[name] = {"parse": "PASS"}
        print(f"[PASS] {name}", flush=True)
    except Exception as e:
        results[name] = {"parse": "FAIL", "error": str(e)}
        print(f"[FAIL] {name}: {e}", flush=True)

Path("capability_matrix.json").write_text(json.dumps(results, indent=2))
print("\nSaved capability_matrix.json")
