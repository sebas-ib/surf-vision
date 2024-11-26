# Surf Vision

Our Team: Jahir, Arturo, Sebastian, Bryce, Zino

**Categorizing Waves with Computer Vision**

As surfers (and soon to be ones), we want to understand wave patterns to improve our & others surfing experience.

AI technique we’ll use: **Convolutional Neural Networks**

**Why Use Convolutional Neural Networks (CNNs)?**
- Capture spatial hierarchies
  - Excellent for identifying patterns and features in waves
- Effective recognizing patterns regardless of position
  - Varying wave positions
- Robust to variations in lighting, angle, and scale 
  - Natural environment pictures
- Lots of research, tools, and libraries available

## Usage

Train and evaluate surf prediction models using the `main.py` script.

### Command Line Arguments

- `--model`: Model to use for prediction (default: `random_forest`).
  - Available models: random_forest, logistic_regression
- `--spot`: Surf spot to use for data (required).
  - Available spots: blacks_beach, cardiff_reef, malibu_north, oceanside_harbour, oceanside_pier, san_onofre

### Example Commands

Train a Random Forest model on data from Blacks Beach:
```python main.py --model random_forest --spot blacks_beach```
