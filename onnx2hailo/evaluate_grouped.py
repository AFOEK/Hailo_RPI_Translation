from spline_probe_grouped import evaluate, OUTPUTS, MODEL, PRECISION

optimized_har = OUTPUTS / f"{MODEL}_{PRECISION}_optimized.har"
metrics_path = OUTPUTS / f"{MODEL}_{PRECISION}_metrics.json"

print("[EVALUATE] Running grouped model separately", flush=True)
evaluate(optimized_har, metrics_path)
print("[DONE] Evaluation successful", flush=True)