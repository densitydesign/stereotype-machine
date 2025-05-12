This is a simplified buide for non-techincal users that want to use the tool without harmign their computer.

## Requirements

- A Mac computer produced after 2021 (you can check by clicking on the apple icon at top right of your computer, select "About this mac" and check if as chip you have "Apple M1" or higher)
- The app [Draw Things](https://drawthings.ai/)
- Python 3.12 or later (you can check by opening the "Terminal" app and typing `python --version`)

## Installation

1. **Download the source cod from GitHub**
   click on the top-right button "Code" on the GitHub page, thenk on "Download Zip"

2. **Unzip the file in a folder**

3. **Drag the folder on the "Terminal" app.**
   A new session of terminal will open

4. **Create a Virtual Environment with Python 3.12:**
   type in terminal:

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

5. **Install Dependencies:**
   type in terminal:

   ```bash
   pip install -r requirements.txt
   ```

6. **Open Draw Things app**

7. **Create custom script**
   Go to the "scripts" panel in the left menu.
   Click on the "+" icon at the top. It will open a window to create a new script. Give the name you prefere, e.g. "Downloader"

8. **Add Custom script**
   click on the new script. It will open a window. Past this code:

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

   and save.

9. **Run the custom script**
   click on the "play button" (the right pointing triangle) near the script.
   It will download and install all the required models.
   It can require a long time depending on your internet connection.

10. **Enable API endpoint:**
    In Draw Things left panel select "settings" and then "all".
    Scroll down to the "API server" section.
    Click on the toggle button to activate it.
    Check that "port" is se to `7860`

11. **Start the tool**
    Go back to the terminal app opened on the stereotype machine folder and type
    ```bash
        python main.py
    ```
