import nncase
from pathlib import Path

# === USER SETTINGS ===
model_path = Path("C:/Users/magic/Downloads/mobilnetv1.onnx")  # 👈 change this
output_path = Path("C:/Users/magic/Downloads/mobilenetv1.kmodel")
target = "k230"
quantize = True

# === LOAD MODEL ===
with open(model_path, "rb") as f:
    content = f.read()

# === COMPILE ===
compiler = nncase.Compiler()
options = nncase.CompileOptions()
options.target = target
options.input_type = "onnx"
options.dump_ir = False
options.dump_dir = "./nncase_dump"
options.quant_type = "uint8" if quantize else "none"
options.preprocess = False

compiler.compile(content, options)

# === SAVE OUTPUT ===
with open(output_path, "wb") as f:
    f.write(compiler.get_model())

print("✅ Compilation complete!")
