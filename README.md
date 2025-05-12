# Stereotype Machine

This tool is aimed at assessing how different nationalities are represented by generative AI (currently using the [Stable Diffusion](https://stability.ai/stable-diffusion) model).

The project was developed within the _[Information Literacy for Societal Resilience](https://tacticaltech.org/news/project-launches/information-literacy-for-societal-resilience/)_ research initiative, a collaboration between [DensityDesign Lab](https://densitydesign.org/) (Politecnico di Milano), [Tactical Tech](https://tacticaltech.org/), and [IFLA](https://www.ifla.org/). The overall goal of the project is to explore how AI is affecting the ways media and information are produced, distributed, and perceived.

This tool was used to produce some of the pieces featured in the exhibition _"Supercharged by AI"_.

The tool uses an _ambiguous prompting_ technique: by providing open-ended prompts (such as _"an Italian family"_) and generating many images with the same prompt, it is possible to observe recurring visual clichés used by the AI model.

The current interface allows users to input a nationality and a main subject (e.g., _family_, _workers_, _teenagers_) to automatically generate a set number of images with the same prompt.

The tool also enables users to create grids from the generated images and to cut out specific features they wish to highlight.

## How to use

The tool at the moment is not self-contained, therefore you must download and execute the code on your machine.

### Requirements

The tool at the moment works only on MacOS. You will need:

- a Mac computer with at least an M1 chip.
- Python 3.12 installed on the machine (usually is alredy installed)
- the app [Draw Things](https://drawthings.ai/)

## Setup

### Simplified guide

For the less experiencd people, you can follow [this simplified guide](simplified-guide.md)

### Advanced guide

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

   **Important:**  
   Create the virtual environment with **Python 3.12** or it will **NOT work**.

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setupping DrawThings:**

Open the drawthings app. The following script utilizes a mixture of Nets and models. To ensure the models are loaded to your drawthings

Go to the "scripts" panel in the left menu

![Script running](ReadMe_images/scripts.png)

Create a new script: "Downloader".
Paste this code:

```js
pipeline.downloadBuiltins([
  "4x_ultrasharp_f16.ckpt",
  "sd_xl_base_1.0_f16.ckpt",
  "sd_xl_refiner_1.0_f16.ckpt",
  "controlnet_tile_1.x_v1.1_f16.ckpt",
  "juggernaut_reborn_q6p_q8p.ckpt",
  "add_more_details__detail_enhancer___tweaker__lora_f16.ckpt",
  "sdxl_render_v2.0_lora_f16.ckpt",
  "tcd_sd_v1.5_lora_f16.ckpt",
]);
```

Run the script.
<img src="ReadMe_images/image-1.png" alt="Script running" width="300"><br><br>

You will now see a downloading interface with the models, controlNets and Loras being downloaded.

6. **Enable API endpoint:**

**NB:** The _API SERVER_ option of Drawthings app must be activated and the port must be set to  
<img src="ReadMe_images/api_server.png" alt="API SERVER" width="300"><br><br> `127.0.0.1:7860` in order to work properly.

5. **Usage:**

After setting up the environment as before, you can start the application as follows:

```bash
python main.py
```

# Application itself

## Image generation

The image generation panel works by defining a queue of operations to be fed to the [Drawthings app](https://apps.apple.com/it/app/draw-things-ai-generation/id6444050820?l=en-GB).

![IMAGE GENERATION WINDOWS](ReadMe_images/1.png)

So just create the queue and launch the generation. On the right window you will be updated with the last image generated.

The nationality and the category will be combined to create a prompt that will look like this:

```python
prompt = f"{Nationality} {Category}, (35mm lens photography), extremely detailed, 4k, shot on dslr, photorealistic, photographic, sharp"
```

Everything that's there except from the Nationality and Category is due to the need of photorealistic outputs.

## Image cutting

## Grid creator

## ToDoList

- [x] Test script on fresh environment
- [ ] Add photoshop file building support
- [x] Check input field open
- [ ] Customize UI?
- [x] Add check for models and force download them if necessary
