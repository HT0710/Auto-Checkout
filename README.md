# Auto-Checkout

## Install
### **Python 3.10**

#### Using CPU
Run `./setup-cpu.sh`

#### Using CUDA
Run `./setup-cuda.sh`

## Usage
### Video folder format
```
.
├── ...
├── modules
└── video
    ├── class_1
    │   ├── video_1.mp4
    │   ├── video_2.mp4
    │   └── ...
    ├── class_2
    │   ├── video_1.mp4
    │   ├── video_2.mp4
    │   └── ...
    ├── class_3
    │   ├── video_1.mp4
    │   ├── video_2.mp4
    │   └── ...
    └── ...
```

### Run
Change configurations in `configs/run.yaml`

```python
python3 run.py
```
