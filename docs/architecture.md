HelioMorph-Edge/
├── README.md
├── requirements.txt
├── setup.sh
├── Dockerfile
├── configs/
│   └── heliomorph.yaml
├── src/
│   ├── models/
│   │   ├── tsve.py
│   │   ├── ecg.py
│   │   ├── tpvml.py
│   │   └── pasr.py
│   ├── prompts/
│   │   └── solar_fault_prompts.json
│   ├── dataset/
│   │   └── solar_dataset.py
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   ├── explain.py
├── edge/
│   ├── export_onnx.py
│   ├── optimize_openvino.py
│   └── trt_build.py
├── deployment/
│   ├── jetson/
│   └── rpi/
├── notebooks/
│   └── exploratory.ipynb
├── scripts/
│   ├── download_data.sh
│   └── preprocess.sh
├── evaluation/
│   └── benchmark.py
└── docs/
    └── architecture.md
