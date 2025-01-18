# EMIF_Reloaded_UI

**Important:**  
Create the virtual environment with **Python 3.12** or it will **NOT display a shit**.

## Overview

EMIF_Reloaded_UI is a user interface project designed to integrate and streamline the image generation, cutting, and grid creation processes. This project builds upon TDL emif concepts and incorporates new features and aesthetic customizations for an improved user experience.

## Features

- **Dynamic UI:**  
  Replace the preset nation selection with an open field to allow free-form input.

- **Model Offloading:**  
  Download models via URL to decrease local storage requirements and facilitate updates.

- **Enhanced Troubleshooting:**  
  Updated troubleshooting procedures for new environment issues.

- **Internal Check-Ins:**  
  Automated readiness checks for models before processing.

- **macOS Specifics:**  
  Custom macOS icon creation to provide a native look and feel.

- **Aesthetic Customization:**  
  Tailor the UI styling and design to meet specific branding or usability requirements.

## Requirements

- **Python:**  
  Python 3.12 is required.  
  _Ensure you create your virtual environment using Python 3.12._

- **Dependencies:**  
  Install all Python package dependencies (listed in `requirements.txt`).

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/EMIF_Reloaded_UI.git
   cd EMIF_Reloaded_UI
   ```

2. **Create a Virtual Environment with Python 3.12:**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After setting up the environment, you can start the application as follows:

```bash
python main.py
```
