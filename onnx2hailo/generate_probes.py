import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from pathlib import Path

OUT = Path("probes")
OUT.mkdir(exist_ok=True)

def save(name, node, inputs, outputs, initializers=[]):
    graph = h.make_graph([node], name, inputs, outputs, initializer=initializers)
    model = h.make_model(graph, producer_name="onnx2hailo",
                         opset_imports=[h.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, OUT / f"{name}.onnx")
    print(f"Generated: {name}")

w = nh.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), name="W")
save("conv", h.make_node("Conv", ["x", "W"], ["y"]),
    [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 8, 8])],
    [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 8, 8])], [w])

table = nh.from_array(np.arange(128, dtype=np.float32).reshape(16, 8), name="table")
save("gather", h.make_node("Gather", ["table", "ids"], ["y"], axis=0),
    [h.make_tensor_value_info("ids", T.INT64, [1, 4])],
    [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 8])], [table])

save("softmax", h.make_node("Softmax", ["x"], ["y"], axis=1),
    [h.make_tensor_value_info("x", T.FLOAT, [1, 4, 8, 8])],
    [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 8, 8])])

proj = nh.from_array(np.eye(8, dtype=np.float32), name="proj")
graph = h.make_graph([
    h.make_node("Gather", ["table", "ids"], ["lookup"], axis=0),
    h.make_node("MatMul", ["lookup", "proj"], ["y"])
], "gather_matmul",
    [h.make_tensor_value_info("ids", T.INT64, [1, 4])],
    [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 8])],
    initializer=[table, proj])
model = h.make_model(graph, opset_imports=[h.make_operatorsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, OUT / "gather_matmul.onnx")