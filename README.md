# SVD Image Compression

Project from Linear Algebra, which aims to implement SVD-based image compression in Python and study how the rank parameter k affects the trade-off between storage reduction and visual quality.

## Installation

Clone the repository:
```bash
git clone [https://github.com/dd8ria/SVD_ImageCompression.git](https://github.com/dd8ria/SVD_ImageCompression.git)
```

## Usage

1. Activate Python virtual environment
   
For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install the libraries from the requirements:

```bash
pip install -r requirements.txt
```

## Project structure

```text
SVD_ImageCompression/
├── images/               # Selected personal photos to reflect different visual characteristics
├── results/              # Output directory for reconstructed images, metrics, and plots
├── svd_utils.py          # Python script with helper functions for image processing
├── svd_experiments.py    # Python script for experiments and Eckart-Young validation
├── SVD_Image_Compression.ipynb # Main notebook containing the implementation pipeline
└── requirements.txt      # Python dependencies
```

## Contributors

* Daryna Nychyporuk
* Anastasiia Yablunovska
* Anastasiia Liubenchuk

### Links to videos

*Liubenchuk Anastasia*: <https://youtu.be/muf06BzSLc4>
