# Automated Meter Reading (YOLO ROI + CNN Digits)

This project performs automated water meter reading with a 2-machine socket pipeline:

- `machine_a/detect_and_send.py`: detects meter counter ROI with YOLO, crops ROI, sends it over socket.
- `machine_b/receive_and_read.py`: receives ROI, segments digits, classifies digits with CNN, and returns final reading.

## Project layout

- `models/roi_model/yolo_model.pt`: YOLO ROI detector weights.
- `models/cnn_model.keras`: CNN digit classifier weights.
- `test_images/`: test images.
- `machine_a/output/`: visual outputs from Machine A.
- `machine_b/debug_outputs/`: segmentation/classification debug outputs from Machine B.

## Requirements

- Python 3.10+ (3.10/3.11 recommended)
- Windows PowerShell (commands below are PowerShell)
- Installed dependencies from `requirements.txt`

## Setup

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the system

Open two terminals in the project root.

### Terminal 1: Start Machine B server

```powershell
.\.venv\Scripts\Activate.ps1
python machine_b\receive_and_read.py --host 127.0.0.1 --port 9999 --model-path models\cnn_model.keras --debug-dir machine_b\debug_outputs\run_local
```

### Terminal 2: Run Machine A client

For a folder of test images:

```powershell
.\.venv\Scripts\Activate.ps1
python machine_a\detect_and_send.py --input test_images --model-path models\roi_model\yolo_model.pt --host 127.0.0.1 --port 9999 --output-dir machine_a\output\run_local --save-roi
```

For a single image:

```powershell
python machine_a\detect_and_send.py --input path\to\meter_image.jpg --output-dir machine_a\output\single_test --save-roi
```

## Expected outputs

- Machine A prints:
  - detected box coordinates
  - method (`yolo` or fallback)
  - confidence
  - final reading returned by Machine B
- Machine A saves:
  - full visualization per image in `machine_a/output/...`
  - optional cropped ROI in `machine_a/output/.../roi_crops`
- Machine B saves (unless `--no-debug-save`):
  - debug panels with ROI, binary segmentation, predicted digit strip in `machine_b/debug_outputs/...`

## Common issues

- `Could not find YOLO model weights`:
  - confirm `models/roi_model/yolo_model.pt` exists.
- `Could not find the CNN model`:
  - confirm `models/cnn_model.keras` exists.
- Port conflict on `9999`:
  - change `--port` on both Machine A and Machine B to the same free port.
- If you run training notebooks and confusion matrix plot fails:
  - in `train_cnn_ufpr_amr.ipynb`, replace `ftfmt='d'` with `fmt='d'` in `sns.heatmap(...)`.

## Notes on the latest CNN run

From notebook outputs in `train_cnn_ufpr_amr.ipynb`:

- Total digit crops: 9,960
- Train/test split: 7,968 / 1,992
- Final reported hold-out accuracy: 99.00%
- EarlyStopping triggered at epoch 124, restoring best epoch 104
