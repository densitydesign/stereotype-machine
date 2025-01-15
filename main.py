import tkinter as tk
from tkinter import ttk
from gui import ImageGenerationApp, ImageCuttingApp, GridCreationApp
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("EMIF Multi-App GUI")

    # Set window dimensions
    screen_width = root.winfo_screenwidth() - 100
    screen_height = root.winfo_screenheight() - 100

    # Create a notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Tab 1: Image Generation App | Per generare le immagini
    tab1 = tk.Frame(notebook)
    notebook.add(tab1, text="Image Generation")
    ImageGenerationApp(tab1)

    # Tab 2: Image Cutting App | Per applicare CV e ritagli
    tab2 = tk.Frame(notebook)
    notebook.add(tab2, text="Image Cutting")
    ImageCuttingApp(tab2)

    # Tab 3: Grid Creation App | Per creare le griglie dopo aver creato le maschere
    tab3 = tk.Frame(notebook)
    notebook.add(tab3, text="Grid Creation")
    GridCreationApp(tab3)

    # Run the Tkinter main loop
    root.mainloop()
