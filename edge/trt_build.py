import argparse
import os

def main(onnx_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Placeholder: integrate TensorRT build using trtexec in practice
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={os.path.join(outdir, 'heliomorph.trt')}"
    print(cmd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    main(args.onnx, args.outdir)
