from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh

out = Path("probes")
out.mkdir(exist_ok=True)

def save(name, nodes, inputs, outputs, weights):
    graph = h.make_graph(nodes, name, inputs, outputs, initializer=weights)
    model = h.make_model(graph, opset_imports=[h.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out / f"{name}.onnx")
    print("Saved:", name)

W = nh.from_array(np.arange(64, dtype=np.float32).reshape(8, 8) / 64, "W")
save("matmul_only",
    [h.make_node("MatMul", ["x", "W"], ["y"])],
    [h.make_tensor_value_info("x", T.FLOAT, [1, 4, 8])],
    [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 8])],
    [W])

E = np.arange(128, dtype=np.float32).reshape(16, 8) / 128
K = nh.from_array(E.T.reshape(8, 16, 1, 1).copy(), "K")
save("onehot_conv",
    [h.make_node("Conv", ["onehot", "K"], ["y"])],
    [h.make_tensor_value_info("onehot", T.FLOAT, [1, 16, 1, 4])],
    [h.make_tensor_value_info("y", T.FLOAT, [1, 8, 1, 4])],
    [K])

vocab = nh.from_array(np.arange(16,dtype=np.float32).reshape(1,16,1,1),"vocab")
one = nh.from_array(np.array(1.,dtype=np.float32),"one")

nodes = [
    h.make_node("Sub",["ids_float","vocab"],["d1"]),
    h.make_node("Sub",["vocab","ids_float"],["d2"]),
    h.make_node("Relu",["d1"],["r1"]),
    h.make_node("Relu",["d2"],["r2"]),
    h.make_node("Add",["r1","r2"],["distance"]),
    h.make_node("Sub",["one","distance"],["peak"]),
    h.make_node("Relu",["peak"],["onehot"]),
    h.make_node("Conv",["onehot","K"],["y"])
]
save("token_id_conv",nodes,
    [h.make_tensor_value_info("ids_float",T.FLOAT,[1,1,1,4])],
    [h.make_tensor_value_info("y",T.FLOAT,[1,8,1,4])],
    [vocab,one,K])