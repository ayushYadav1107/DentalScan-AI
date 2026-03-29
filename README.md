# 🦷 DentalScan AI

**A dental X-ray analysis tool powered by a fine-tuned YOLOv10s model.**  
Upload any dental X-ray image and the model will automatically detect and classify dental conditions — drawing colour-coded bounding boxes directly on the image with confidence scores.

> 🌐 **Don't want to run it locally?**  
> The app is hosted online — just open the link below and use it directly in your browser. No installation, no setup required.
>
> **👉 [Open DentalScan AI — Live Demo](#)** *(replace `#` with your Streamlit / Hugging Face URL once deployed)*

---

## 📋 Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [Detected Classes](#-detected-classes)
3. [Project Structure](#-project-structure)
4. [Requirements](#-requirements)
5. [Setup & Installation](#-setup--installation)
6. [Running the Web App](#-running-the-web-app)
7. [Using the App](#-using-the-app)
8. [Training the Model (Optional)](#-training-the-model-optional)
9. [Evaluating the Model (Optional)](#-evaluating-the-model-optional)
10. [Deploying to the Cloud](#-deploying-to-the-cloud)
11. [Troubleshooting](#-troubleshooting)

---

## 🔍 What This Project Does

DentalScan AI is a **computer vision web application** that analyses dental panoramic or periapical X-ray images. It uses a **YOLOv10s** object detection model that was fine-tuned for 250 epochs on a labelled dental X-ray dataset.

When you upload an X-ray, the app will:
- Detect the location of dental abnormalities and regions of interest
- Draw a colour-coded bounding box around each finding
- Label each box with the condition name and confidence score
- Show a summary of total detections, unique classes found, and average confidence

The interface is a **Streamlit web app** — it runs locally in your browser. No internet connection is needed after setup (except for the initial library download).

---

## 🦠 Detected Classes

The model is trained to detect **6 dental conditions**, each assigned a unique colour:

| # | Class | Description |
|---|-------|-------------|
| 0 | **Caries** | Tooth decay / cavities |
| 1 | **Infection** | Periapical or periodontal infection |
| 2 | **Impacted** | Impacted tooth (did not fully erupt) |
| 3 | **BDC/BDR** | Bone Defect Coronal / Bone Defect Root |
| 4 | **Fractured** | Cracked or broken tooth |
| 5 | **Healthy** | Normal, healthy tooth region |

---

## 📁 Project Structure

```
DentalScan AI/
│
├── .git/                   ← Git repository tracking
├── .gitignore              ← Files/folders ignored by Git
├── app.py                  ← Main Streamlit web app (run this)
├── model.py                ← Script used to train the YOLOv10s model
├── evaluate.py             ← Basic YOLO evaluation script
├── evaluate_new.py         ← Detailed evaluation with per-detection metrics
├── data.yaml               ← Dataset configuration (class names + paths)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── runs/
│   ├── detect/             ← Detection training & validation experiment runs
│   │   ├── Yolo_10s_train/ ← ✅ Main fine-tuned model run (used by the app)
│   │   │   └── weights/
│   │   │       ├── best.pt ← Best checkpoint (lowest validation loss)
│   │   │       └── last.pt ← Last checkpoint
│   │   ├── Yolo_10s_val/
│   │   ├── Yolo_12m_250epochs/
│   │   ├── Yolo_8n_250/
│   │   └── previous/
│   └── segment/            ← Segmentation experiment runs (not used by the app)
│
└── .venv/                  ← Python virtual environment (created during setup)
```

---

## ✅ Requirements

### System Requirements

| Component | Minimum |
|-----------|---------|
| OS | Windows 10/11, macOS, or Linux |
| Python | **3.10 or later** |
| RAM | 8 GB |
| GPU | Optional (NVIDIA CUDA GPU speeds up inference significantly) |
| Disk space | ~2 GB (model weights + dependencies) |

### Python Dependencies

All dependencies are listed in `requirements.txt`:

```
streamlit
ultralytics
Pillow
numpy
opencv-python
```

---

## 🛠 Setup & Installation

Follow these steps **exactly once** to set up the project. After this, you only need to do [Step 5](#step-5-activate-the-virtual-environment) onwards each time.

### Step 1 — Clone or download the project

If you have Git installed:
```bash
git clone <your-repo-url>
cd "DentalScan AI"
```

Or simply download the project as a ZIP and extract it, then open a terminal inside the extracted folder.

---

### Step 2 — Check your Python version

```bash
python --version
```

You should see `Python 3.10.x` or higher. If not, download Python from [python.org](https://www.python.org/downloads/).

---

### Step 3 — Create a virtual environment

A virtual environment keeps project dependencies isolated from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

**macOS / Linux:**
```bash
python3 -m venv .venv
```

---

### Step 4 — Activate the virtual environment

> ⚠️ You must activate the virtual environment **every time** you open a new terminal session.

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

When activated, your terminal prompt will show `(.venv)` at the beginning — for example:
```
(.venv) PS C:\Dev\Python\DentalScan AI>
```

---

### Step 5 — Install dependencies

With the virtual environment active, run:

```bash
pip install -r requirements.txt
```

This installs Streamlit, Ultralytics (YOLOv10), OpenCV, Pillow, and NumPy. It may take a few minutes on first run.

---

### Step 6 — Verify the fine-tuned model weights exist

The app uses the fine-tuned model located at:
```
runs/detect/Yolo_10s_train/weights/best.pt
```

Make sure this file exists before running the app. If it is missing:
- Either re-train the model (see [Training the Model](#-training-the-model-optional))
- Or contact the project owner for the `best.pt` file
- Or use the **hosted demo link** at the top of this README — no local weights needed

> ⚠️ **Important:** If `best.pt` is missing, the app will show an amber warning dot and produce incorrect detections. The base `yolov10s.pt` file is **not** required to run the app — only `best.pt` is needed.

---

## 🚀 Running the Web App

Make sure your virtual environment is activated (you should see `(.venv)` in your terminal), then run:

```bash
streamlit run app.py
```

After a moment, Streamlit will print something like:

```
You can now view your Streamlit app in your browser.
Local URL:  http://localhost:8501
```

Your default browser will open automatically. If it does not, copy and paste `http://localhost:8501` into your browser.

To stop the app, press `Ctrl + C` in the terminal.

---

## 🖥 Using the App

Once the app is open in your browser:

### 1. Check the model status bar
At the top of the page, a status bar shows which model is loaded:
- 🟢 **Green dot** — the fine-tuned model is loaded. Detections will be accurate.
- 🟡 **Amber dot** — the base model is loaded. Detections will be incorrect. See [Troubleshooting](#-troubleshooting).

### 2. Upload an X-ray image
In the **left panel**, click **Browse files** (or drag and drop) to upload a dental X-ray.  
Supported formats: `.jpg`, `.jpeg`, `.png`

### 3. Adjust detection settings (optional)
Two sliders let you tune how detections work:

| Slider | What it does | Tip |
|--------|-------------|-----|
| **Confidence threshold** | Minimum confidence for a box to be shown | Lower it (e.g. `0.10`) if nothing is detected |
| **IoU / NMS threshold** | How much boxes can overlap before one is removed | Lower it to reduce duplicate boxes |

### 4. Click "Run Detection"
Switch to the **🎯 Detection Result** tab, then click the **🔍 Run Detection** button.

The app will:
- Run the YOLOv10s model on your image
- Draw colour-coded bounding boxes on the result
- Show summary metrics (number of detections, classes found, average confidence, inference time)
- List every finding with its class name, bounding box coordinates, and confidence score

### 5. View the class legend
The bottom of the left panel shows the colour corresponding to each of the 6 classes.

---

## 🧪 Training the Model (Optional)

> Only needed if you want to train your own model from scratch or re-train with new data.

**Prerequisites:**
- An NVIDIA CUDA-capable GPU is strongly recommended (training on CPU will take many hours)
- Download the dataset here: **[Dental OPG XRAY Dataset](https://data.mendeley.com/datasets/c4hhrkxytw/4)** and structure it in YOLO format.

**Run training:**
```bash
python model.py
```

The script trains YOLOv10s for 250 epochs using the settings defined in `model.py`. Training results and weights are saved automatically to `runs/detect/Yolo_10s_train/`.

Training configuration used:

| Parameter | Value |
|-----------|-------|
| Base model | `yolov10s.pt` |
| Epochs | 250 |
| Image size | 640 × 640 |
| Batch size | 8 |
| Optimizer | Auto |
| Device | CUDA (GPU) |
| Augmentation | Mosaic, MixUp, Copy-Paste, Flips, Rotation |
| Early stopping patience | 100 epochs |

---

## 📊 Evaluating the Model (Optional)

To run a validation pass on the test set and print precision, recall, and mAP metrics:

```bash
python evaluate.py
```

For a more detailed evaluation that includes per-detection CNN cross-checking:
```bash
python evaluate_new.py
```

---

## ☁️ Deploying to the Cloud

To share the app publicly without anyone needing to install anything:

### Option A — Streamlit Community Cloud (Easiest, Free)

1. Push the entire project folder to a **GitHub repository**.  
   Make sure to include `app.py`, `requirements.txt`, and `runs/detect/Yolo_10s_train/weights/best.pt`.

2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.

3. Click **New App**, select your repository and branch.

4. Set **Main file path** to `app.py`.

5. Click **Deploy**. Streamlit installs dependencies automatically and gives you a public URL like:  
   `https://your-app-name.streamlit.app`

> ⚠️ Note: The free tier has memory limits (~1 GB). The fine-tuned model file (`best.pt` ≈ 16 MB) must be committed to the repository or hosted separately.

---

### Option B — Hugging Face Spaces (Great for ML models, Free)

1. Create a free account at [huggingface.co](https://huggingface.co/).

2. Click **New Space**, name it (e.g. `DentalScan-AI`), and choose **Streamlit** as the SDK.

3. Upload `app.py`, `requirements.txt`, and `runs/detect/Yolo_10s_train/weights/best.pt` using the web interface or Git.

4. Hugging Face will build and host your app automatically.

---

## 🔧 Troubleshooting

### Nothing is detected / wrong class names shown

**Cause:** The fine-tuned model is not found and the app fell back to the base `yolov10s.pt`.

**Fix:** Make sure the file `runs/detect/Yolo_10s_train/weights/best.pt` exists relative to the project folder. The model bar at the top of the app will show an amber dot if it is missing.

---

### `activate.ps1` is not recognized / cannot be loaded

**Cause:** PowerShell execution policy is blocking script execution.

**Fix:** Run this command once in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again:
```powershell
.\.venv\Scripts\activate.ps1
```

---

### `venv\Scripts\activate` not found

**Cause:** You may be using the wrong folder name. This project uses `.venv` (with a dot).

**Fix:** Always use:
```powershell
.\.venv\Scripts\activate.ps1
```
Note the leading `.\` and the `.venv` folder name.

---

### `streamlit: command not found`

**Cause:** The virtual environment is not activated, or dependencies were not installed.

**Fix:**
```bash
# 1. Activate the venv first
.\.venv\Scripts\activate.ps1

# 2. Then install requirements
pip install -r requirements.txt

# 3. Then run the app
streamlit run app.py
```

---

### App is slow / inference takes a long time

**Cause:** The model is running on CPU instead of GPU.

**Fix:** If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then restart the app. Ultralytics will automatically use the GPU.

---

## 📄 License

This project is for academic and research purposes.

---

## 👤 Author

Built with [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10) and [Streamlit](https://streamlit.io/).
