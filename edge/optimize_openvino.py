import argparse
import os
from openvino.runtime import Core

def main(onnx_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    core = Core()
    model = core.read_model(onnx_path)
    compiled = core.compile_model(model, "CPU")
    core.serialize(model, os.path.join(outdir, "model.xml"), os.path.join(outdir, "model.bin"))
    print("Exported OpenVINO IR to", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    main(args.onnx, args.outdir)
