# Dental X-Ray Enhancer

A GUI application for enhancing dental X-ray images using various image processing techniques. The application implements multiple advanced algorithms for denoising, edge enhancement, contrast adjustment, and other image quality improvements specifically designed for dental radiography.

## Installation

### Prerequisites
- Python 3.6 or higher

Choose the installation instructions for your operating system:

## macOS Installation

### Step 1: Install Python
If you don't already have Python installed:
```bash
brew install python
```

### Step 2: Create a virtual environment (recommended)
```bash
python -m venv dental-xray-env
source dental-xray-env/bin/activate
```

### Step 3: Install required packages
```bash
pip install numpy opencv-python scikit-image PyQt5
```

### macOS Troubleshooting

#### Qt issues on macOS
Sometimes PyQt5 has compatibility issues with newer macOS versions. If you encounter problems, try:
```bash
pip uninstall PyQt5
pip install PyQt5-Qt5
pip install PyQt5
```

#### OpenCV issues
If you encounter OpenCV-related errors:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Permission issues
If you get permission errors when running the GUI:
```bash
chmod +x dental_xray_enhancer_pipeline_gui.py
```

#### M1/M2 Mac users
If you're using an Apple Silicon Mac (M1/M2), you might need to use Rosetta for some packages:
```bash
arch -x86_64 pip install PyQt5
```

Or consider using Miniforge/Conda which has better native ARM support:
```bash
brew install miniforge
conda create -n dental-env python=3.9
conda activate dental-env
conda install numpy opencv scikit-image
pip install PyQt5
```

## Linux Installation

### Step 1: Install Python and dependencies
For Debian/Ubuntu-based distributions:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
sudo apt install python3-opencv python3-qt5
```

For Fedora/RHEL-based distributions:
```bash
sudo dnf install python3 python3-pip
sudo dnf install python3-opencv python3-qt5
```

### Step 2: Create a virtual environment
```bash
python3 -m venv dental-xray-env
source dental-xray-env/bin/activate
```

### Step 3: Install required packages
```bash
pip install numpy scikit-image opencv-python PyQt5
```

### Linux Troubleshooting

#### Display issues
If you encounter issues with the display:
```bash
export QT_X11_NO_MITSHM=1
```

#### Missing shared libraries
If you get errors about missing shared libraries:
```bash
sudo apt install libxcb-xinerama0
# Or for Fedora/RHEL
sudo dnf install libxcb-xinerama0
```

#### Permission issues
Make the script executable:
```bash
chmod +x dental_xray_enhancer_pipeline_gui.py
```

## Windows Installation

### Step 1: Install Python
1. Download Python from [python.org](https://www.python.org/downloads/windows/)
2. During installation, check "Add Python to PATH"
3. Choose "Customize installation" and ensure pip is selected

### Step 2: Create a virtual environment
Open Command Prompt as Administrator:
```cmd
python -m venv dental-xray-env
dental-xray-env\Scripts\activate
```

### Step 3: Install required packages
```cmd
pip install numpy opencv-python scikit-image PyQt5
```

### Windows Troubleshooting

#### PyQt5 installation issues
If you encounter issues installing PyQt5:
```cmd
pip install PyQt5-tools
```

#### OpenCV issues
If you get DLL errors when importing OpenCV:
```cmd
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Permission errors
Run Command Prompt as Administrator when executing the script.

## Usage

Navigate to the directory containing the script and run it:

On macOS/Linux:
```bash
python dental_xray_enhancer_pipeline_gui.py
```

On Windows:
```cmd
python dental_xray_enhancer_pipeline_gui.py
```

The application provides a GUI interface for enhancing dental X-ray images with various processing techniques.

## Features

The application provides a comprehensive suite of image processing algorithms to enhance dental X-ray images:

- **Denoising Techniques**:
  - Non-Local Means Denoising - Advanced algorithm that preserves details while removing noise
  - Bilateral Filter - Edge-preserving smoothing filter
  - Median Blur - Removes salt-and-pepper noise
  - Total Variation Denoising - Preserves edges while smoothing noise
  - Edge-Aware Smoothing - Reduces noise while preserving important anatomical edges

- **Enhancement Techniques**:
  - Adjust Contrast & Brightness - Basic image quality improvements
  - Gamma Correction - Adjusts image luminance
  - CLAHE Enhancement - Contrast Limited Adaptive Histogram Equalization for improved local contrast
  - Edge Enhancement - Highlights dental structures
  - Unsharp Mask - Sharpens details in the image
  - Advanced Edge Enhancement - Specialized algorithm for dental structure visualization
  - Overlay Original Image - Compare processed results with original
  - Invert image with strength changable

- Additional features:
  - Image loading and saving
  - Pipeline loading and saving
  - Real-time preview of changes
  - Customizable processing pipeline

