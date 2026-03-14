#!/usr/bin/env python3
"""Generate BatchNormalization.onnx test model and reference output."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Input shape: [1, 2, 3, 2] -> batch=1, channels=2, H=3, W=2 -> 12 elements
X_shape = [1, 2, 3, 2]
channels = X_shape[1]
epsilon = 1e-05

# Per-channel parameters
scale = np.array([2.0, 0.5], dtype=np.float32)
bias = np.array([1.0, -1.0], dtype=np.float32)
mean = np.array([0.0, 1.0], dtype=np.float32)
var = np.array([1.0, 4.0], dtype=np.float32)

# Build ONNX graph
X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, X_shape)
Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, X_shape)

bn_node = helper.make_node(
    "BatchNormalization",
    inputs=["X", "scale", "bias", "mean", "var"],
    outputs=["Y"],
    epsilon=epsilon,
)

graph = helper.make_graph(
    [bn_node],
    "BatchNormTest",
    [X_info],
    [Y_info],
    initializer=[
        numpy_helper.from_array(scale, name="scale"),
        numpy_helper.from_array(bias, name="bias"),
        numpy_helper.from_array(mean, name="mean"),
        numpy_helper.from_array(var, name="var"),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
onnx.checker.check_model(model)
onnx.save(model, "input_models/BatchNormalization.onnx")
print("Saved input_models/BatchNormalization.onnx")

# Compute reference output: Y = scale * (X - mean) / sqrt(var + eps) + bias
x_data = np.array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    dtype=np.float32,
).reshape(X_shape)

y_data = np.empty_like(x_data)
for c in range(channels):
    inv_std = 1.0 / np.sqrt(var[c] + epsilon)
    y_data[0, c] = scale[c] * (x_data[0, c] - mean[c]) * inv_std + bias[c]

vals = y_data.flatten()
print("Input: ", x_data.flatten().tolist())
print("Output:", vals.tolist())

# Write reference header
with open("input_models/references/BatchNormalization.ref.hxx", "w") as f:
    f.write("namespace BatchNormalization_ExpectedOutput{\n")
    f.write("\tfloat outputs[] = {\n")
    lines = []
    for i in range(0, len(vals), 6):
        chunk = vals[i : i + 6]
        lines.append("     " + ", ".join(f"{v:.6f}f" for v in chunk))
    f.write(",\n".join(lines))
    f.write("\n\t};\n")
    f.write("} // namespace BatchNormalization_ExpectedOutput\n")

print("Saved input_models/references/BatchNormalization.ref.hxx")
