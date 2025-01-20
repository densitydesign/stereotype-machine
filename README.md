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
   git clone https://github.com/yourusername/EMIF_2.0.git
   cd EMIF_2.0
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

# Application itself

## Image generation

The image generation panel works by defining a queue of operations to be fed to the [Drawthings app](https://apps.apple.com/it/app/draw-things-ai-generation/id6444050820?l=en-GB).

**NB:** The _API SERVER_ option of Drawthings app must be activated and the port must be set to  
<img src="ReadMe_images/api_server.png" alt="API SERVER" width="300"><br><br> `127.0.0.1:7860` in order to work properly.

Then, all the configurations and model will be automatically downloaded by the script.

![IMAGE GENERATION WINDOWS](ReadMe_images/1.png)

## Image cutting

## Grid creator

## ToDoList

- [ ] Test script on fresh environment
- [ ] Add photoshop file building support
- [ ] Check input field open
- [ ] Customize UI?
- [ ] Add check for models and force download them if necessary
