import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from image_generation.generate_pipeline import generate_images, upscale_image
from image_generation.variable_set import variable_sets
import queue
import sys
import tkinter.filedialog as fd
import os
from image_cutting_sam2 import process_images, initialize_sam_model
from create_grids import create_image_grids, find_images
from PIL import Image, ImageTk
import queue
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
folder_path = os.path.join(project_root, "Output_Folder")

logging.basicConfig(
    filename="gui_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_widget_info(widget, name="Widget"):
    """Log widget's geometry and configuration information."""
    try:
        logging.debug(f"{name}: {widget.winfo_class()} at {widget.winfo_geometry()}")
    except Exception as e:
        logging.error(f"Error logging {name}: {e}")

class DebugMixin:
    """Mixin class to add debug logging for GUI components."""

    def debug_log(self, message):
        logging.debug(message)

class ImageGenerationApp:
    
    def __init__(self, frame):
        logging.debug("Initializing ImageGenerationApp")
        self.frame = frame

        # Folder path variables
        self.default_generated_folder = os.path.join(project_root, "Generated_Images")
        self.default_upscaled_folder = os.path.join(project_root, "Upscaled_Images")
        
        os.makedirs(self.default_generated_folder, exist_ok=True)
        os.makedirs(self.default_upscaled_folder, exist_ok=True)

        self.generated_folder = tk.StringVar(value=self.default_generated_folder)
        self.upscaled_folder = tk.StringVar(value=self.default_upscaled_folder)

        # Animation variables
        self.animation_chars = "|/-\\"
        self.animation_index = 0
        self.animation_running = False  # Initialize animation_running

        # Queue and progress tracking
        self.operation_queue = []  # Pending queue
        self.processing_tasks = []  # Processing queue
        self.error_log = queue.Queue()  # Queue for error log messages
        self.progress_info = {
            "time_spent": 0,
            "estimated_time_left": "Not yet defined",
            "progress_percent": 0,
        }
        self.stop_flag = False
        self.first_image_generated = False  # Track if the first image is done

        self.steps_var = tk.StringVar(value="30")
        self.steps_options = [str(i) for i in range(10, 101, 10)]

        try:
            self._setup_gui()
            logging.debug("GUI setup completed")
        except Exception as e:
            logging.error(f"Error during GUI setup: {e}")

    def display_image(self, image_path):
        """
        Display the last generated image in the UI, ensuring that it fits within the container.
        
        Args:
            image_path (str): Path to the image to display.
        """
        try:
            logging.debug(f"Attempting to display image: {image_path}")
            # Open the image
            img = Image.open(image_path)
            
            # Obtain container dimensions from the image label
            container_width = self.image_label.winfo_width()
            container_height = self.image_label.winfo_height()
            
            # If the container dimensions are not yet available (or too small), use defaults
            if container_width < 10 or container_height < 10:
                container_width, container_height = 700, 700
            
            # Resize the image to fit within the container, preserving the aspect ratio.
            # The thumbnail() method modifies the image in place.
            img.thumbnail((container_width, container_height), Image.Resampling.LANCZOS)
            
            # Convert the image to a format Tkinter can display
            photo = ImageTk.PhotoImage(img)
            
            # Update the label with the new image and clear any placeholder text.
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference to avoid garbage collection
            logging.info(f"Image displayed: {image_path}")

        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            messagebox.showerror("Error", f"Image file not found: {image_path}")
        except Exception as e:
            logging.error(f"Error displaying image: {e}")
            messagebox.showerror("Error", f"Error displaying image: {e}")

    def _setup_gui(self):
        # Set custom font globally for all widgets
        try:
            logging.debug("Setting up GUI components")
            self.frame.option_add("*Font", "InstrumentSans 12")
        except tk.TclError:
            self.frame.option_add("*Font", "Helvetica 12")

        # Main Title Label
        title_label = tk.Label(self.frame, text="Image Generation App", font=("InstrumentSans", 16, "bold"))
        title_label.pack(pady=10)

        # Description Label
        description_label = tk.Label(
            self.frame,
            text=(
                "🎨 This section allows you to generate the images."
                "\n 📋 You can create a queue of generations."
                "\n 🛜 Ensure the HTTP-server is active in DrawThings GUI"
            ),
            wraplength=500,  # Set a maximum width for the text
            justify="left",  # Align the text to the left
        )
        description_label.pack(pady=10)

        image_frame = tk.Frame(self.frame)
        image_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame, text="No Image Yet", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Selection Frame
        selection_frame = tk.Frame(self.frame)
        selection_frame.pack(pady=10)

        # Nation selection with note and default value
        nation_note = tk.Label(selection_frame, text="Write here the nationality of what you want to generate:", font=("InstrumentSans", 8), fg="gray")
        nation_note.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

        nation_label = tk.Label(selection_frame, text="Enter Nationality:")
        nation_label.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.nation_var = tk.StringVar(value="Italian")  # Pre-inserted default value (modify as needed)
        nation_entry = tk.Entry(selection_frame, textvariable=self.nation_var)
        nation_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Category selection with note and default value
        category_note = tk.Label(selection_frame, text="Write here the category you want to generate:", font=("InstrumentSans", 8), fg="gray")
        category_note.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

        category_label = tk.Label(selection_frame, text="Enter Category:")
        category_label.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.category_var = tk.StringVar(value="Family")  # Pre-inserted default value (modify as needed)
        category_entry = tk.Entry(selection_frame, textvariable=self.category_var)
        category_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # Number of images
        num_images_label = tk.Label(selection_frame, text="Number of Images:")
        num_images_label.grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.num_images_var = tk.StringVar(value="1")
        num_images_entry = tk.Entry(selection_frame, textvariable=self.num_images_var)
        num_images_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Number of Steps selection remains unchanged
        steps_label = tk.Label(selection_frame, text="Number of Steps:")
        steps_label.grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.steps_combobox = ttk.Combobox(selection_frame, textvariable=self.steps_var, values=self.steps_options, state="readonly")
        self.steps_combobox.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        # Add to Queue button
        add_to_queue_button = tk.Button(selection_frame, text="Add to Queue", command=self.add_to_queue)
        add_to_queue_button.grid(row=6, column=0, columnspan=2, pady=10)


        # Folder selection for generated images
        folder_frame = tk.Frame(self.frame)
        folder_frame.pack(pady=10)

        # Folder selection for generated images
        gen_folder_label = tk.Label(folder_frame, text="📁 Generated Images Folder:")
        gen_folder_label.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        gen_folder_button = tk.Button(folder_frame, text="Browse...", command=self.select_generated_folder)
        gen_folder_button.grid(row=0, column=1, padx=5, pady=5)

        self.generated_folder_label = tk.Label(folder_frame, textvariable=self.generated_folder, width=40, anchor="w")
        self.generated_folder_label.grid(row=0, column=2, padx=5, pady=5)


        # Folder selection for upscaled images
        upscale_folder_label = tk.Label(folder_frame, text="📁 Upscaled Images Folder:")
        upscale_folder_label.grid(row=1, column=0, sticky="e", padx=5, pady=5)

        upscale_folder_button = tk.Button(folder_frame, text="Browse...", command=self.select_upscaled_folder)
        upscale_folder_button.grid(row=1, column=1, padx=5, pady=5)

        self.upscaled_folder_label = tk.Label(folder_frame, textvariable=self.upscaled_folder, width=40, anchor="w")
        self.upscaled_folder_label.grid(row=1, column=2, padx=5, pady=5)


        # Queue Frame
        queue_frame = tk.Frame(self.frame)
        queue_frame.pack(pady=10)

        # Left column: Pending Queue
        left_queue_label = tk.Label(queue_frame, text="Pending Queue:")
        left_queue_label.grid(row=0, column=0, padx=5, pady=5)

        self.queue_listbox = tk.Listbox(queue_frame, width=30, selectmode=tk.SINGLE)
        self.queue_listbox.grid(row=1, column=0, padx=5, pady=5)


        # Right column: Processing Queue
        right_queue_label = tk.Label(queue_frame, text="Processing Queue:")
        right_queue_label.grid(row=0, column=1, padx=5, pady=5)

        self.processing_listbox = tk.Listbox(queue_frame, width=30)
        self.processing_listbox.grid(row=1, column=1, padx=5, pady=5)

        # Start, Stop, and Remove Buttons
        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Image Generation", command=self.start_image_generation)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.confirm_stop, bg="red", fg="black")
        self.stop_button.grid(row=0, column=1, padx=5)

        remove_button = tk.Button(button_frame, text="Remove from Queue", command=self.remove_from_queue)
        remove_button.grid(row=0, column=2, padx=5)

        # Error Log Frame
        error_log_frame = tk.Frame(self.frame)
        error_log_frame.pack(pady=10)
        error_log_label = tk.Label(error_log_frame, text="Error Log:")
        error_log_label.pack()
        self.error_log_text = tk.Text(error_log_frame, height=5, width=50, state=tk.DISABLED)
        self.error_log_text.pack(padx=5, pady=5)


        # Idle Animation Label
        self.animation_label = tk.Label(self.frame, text="", font=("InstrumentSans", 12))
        self.animation_label.pack(pady=5)

        # Progress Frame
        progress_frame = tk.Frame(self.frame)
        progress_frame.pack(pady=10)
        progress_label = tk.Label(progress_frame, text="Progress:")
        progress_label.pack()
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10)
        self.time_spent_label = tk.Label(progress_frame, text="Time Spent on Current Image: 0s")
        self.time_spent_label.pack()
        self.estimated_time_label = tk.Label(progress_frame, text="Estimated Time Left: Not yet defined")
        self.estimated_time_label.pack()

        # Queue Listbox with drag-and-drop functionality
        self.queue_listbox.bind("<Button-1>", self.select_item)
        self.queue_listbox.bind("<B1-Motion>", self.drag_item)

        # Bind window close event to on_close method

        # Periodic progress, animation, and error log updates
        self.update_progress()
        self.update_animation()
        self.update_error_log()  # Start error log update
        
    def on_close(self):
        # Set stop flag to terminate any ongoing processes
        self.stop_flag = True
        
        # Wait briefly to ensure threads have time to terminate
        time.sleep(0.2)

        # Destroy the Tkinter window and exit the application
        self.frame.destroy()
        print("Application closed.")
        sys.exit()  # Ensures command line stops

    def select_item(self, event):
        # Track the item selected for dragging
        self.selected_index = self.queue_listbox.nearest(event.y)

    def drag_item(self, event):
        # Detect the index to move the item to and update list order
        target_index = self.queue_listbox.nearest(event.y)
        if target_index != self.selected_index and target_index >= 0:
            # Swap in Listbox
            temp = self.queue_listbox.get(self.selected_index)
            self.queue_listbox.delete(self.selected_index)
            self.queue_listbox.insert(target_index, temp)

            # Update operation_queue
            self.operation_queue.insert(target_index, self.operation_queue.pop(self.selected_index))
            self.selected_index = target_index

    def select_generated_folder(self):
        folder_path = fd.askdirectory(title="Select Folder for Generated Images")
        if folder_path:
            self.generated_folder.set(folder_path)

    def select_upscaled_folder(self):
        folder_path = fd.askdirectory(title="Select Folder for Upscaled Images")
        
        if folder_path:
            self.upscaled_folder.set(folder_path)

        return folder_path

    def add_to_queue(self):
        chosen_nation = self.nation_var.get()
        category = self.category_var.get()
        steps = int(self.steps_var.get())
        num_images_str = self.num_images_var.get()
        
        
        if not chosen_nation:
            messagebox.showerror("Error", "Please select a nation.")
            return
        if not category:
            messagebox.showerror("Error", "Please select a category.")
            return
        if not num_images_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid number of images.")
            return
        num_images = int(num_images_str)

        task = (chosen_nation, category, num_images, steps)
        self.operation_queue.append(task)
        self.queue_listbox.insert(tk.END, f"Nation: {chosen_nation} - Category: {category} - Number: {num_images} - Steps: {steps}")

    def start_image_generation(self):
        if not self.operation_queue:
            messagebox.showerror("Error", "Operation queue is empty.")
            return
        if self.generated_folder.get() == "Select a folder" or self.upscaled_folder.get() == "Select a folder":
            messagebox.showerror("Error", "Please select both folders before starting.")
            return

        self.stop_flag = False
        self.first_image_generated = False
        self.start_button.config(state="disabled")
        self.animation_running = True  # Start the animation
        threading.Thread(target=self.process_queue).start()

    def confirm_stop(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the image generation?"):
            self.stop_image_generation()

    def stop_image_generation(self):
        self.stop_flag = True
        self.animation_running = False  # Stop the animation
        self.progress_info["estimated_time_left"] = "Not yet defined"  # Reset estimated time
        self.progress_info["time_spent"] = 0  # Reset time spent on the current image
        self.time_spent_label.config(text="Time Spent on Current Image: 0s")  # Reset label
        self.estimated_time_label.config(text="Estimated Time Left: Not yet defined")  # Reset label
        print("Image generation process stopped.")

    def process_queue(self):
        total_images = sum(num_images for _, _, num_images, _ in self.operation_queue)
        images_processed = 0
        time_per_image = []

        while self.operation_queue and not self.stop_flag:

            task_start_time = time.time()

            def update_timer():
                if not self.stop_flag and self.operation_queue:
                    elapsed_time = time.time() - task_start_time
                    formatted_time = self.format_time(elapsed_time)
                    self.time_spent_label.config(text=f"Time Spent on Current Image: {formatted_time}")
                    self.frame.after(500, update_timer)

            update_timer()

            task = self.operation_queue[0]
            chosen_nation, category, num_images, steps = task
            self.queue_listbox.delete(0)
            self.processing_listbox.insert(tk.END, f"{chosen_nation} - {category} - Remaining: {num_images - 1}")
            self.processing_tasks.append((chosen_nation, category, 1)) 

            try:
                # Display the dummy image as a placeholder
                dummy_image_path = f"{project_root}/BG_dummy.png"
                if os.path.exists(dummy_image_path):
                    self.display_image(dummy_image_path)
                else:
                    self.error_log.put(f"Dummy image not found: {dummy_image_path}")

                # Call the generate_images function and capture the path of the generated image
                last_image_path = generate_images(
                    chosen_nation, 
                    category, 
                    1, 
                    self.generated_folder.get(), 
                    self.upscaled_folder.get(),
                    steps
                )

                # Display the image returned by generate_images
                if os.path.exists(last_image_path):
                    self.display_image(last_image_path)
                else:
                    self.error_log.put(f"Image not found: {last_image_path}")
            except Exception as e:
                self.error_log.put(f"Error generating or displaying image: {str(e)}")



            # Calculate time spent on the current image
            time_spent = time.time() - task_start_time
            time_per_image.append(time_spent)  # Add the time spent to our tracking list

            # Update counters and remaining images
            images_processed += 1
            num_images -= 1  # Reduce count for remaining images in this task

            # Calculate average time per image and estimated time remaining in minutes
            avg_time_per_image = sum(time_per_image) / len(time_per_image)
            remaining_images = sum(num_images for _, _, num_images, _ in self.operation_queue) + len(self.processing_tasks) - images_processed
            estimated_time_left = (avg_time_per_image * remaining_images) / 60  # Convert to minutes

            # Update progress info with new values
            self.progress_info["time_spent"] = time_spent
            self.progress_info["estimated_time_left"] = f"{estimated_time_left:.2f} min"
            self.progress_info["progress_percent"] = (images_processed / total_images) * 100

            # Update or remove task based on remaining images
            if num_images > 0:
                self.operation_queue[0] = (chosen_nation, category, num_images, steps)
            else:
                self.operation_queue.pop(0)

            # After task completes, remove from "Being Processed"
            self.processing_listbox.delete(0)
            self.processing_tasks.pop(0)

        # Reset Start button and stop flag after processing
        self.start_button.config(state="normal")
        self.stop_flag = False
        
        # Show completion notification
        messagebox.showinfo("Image Generation Complete", "All images have been successfully created!")
        print("Image generation process complete.")

    def remove_from_queue(self):
        selected_index = self.queue_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            self.queue_listbox.delete(index)
            del self.operation_queue[index]
        else:
            messagebox.showerror("Error", "No item selected to remove from the queue.")

    def format_time(self, seconds):
        if seconds >= 86400:  # More than a day
            days = int(seconds // 86400)
            return f"{days} day{'s' if days > 1 else ''}"
        elif seconds >= 3600:  # More than an hour
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours} hr{'s' if hours > 1 else ''} {minutes} min"
        elif seconds >= 60:  # More than a minute
            minutes = int(seconds // 60)
            sec = int(seconds % 60)
            return f"{minutes} min {sec} sec"
        else:
            return f"{int(seconds)} sec"

    def update_progress(self):
        # Update progress bar and formatted time
        self.progress_var.set(self.progress_info.get("progress_percent", 0))

        # Format and display elapsed time on current image
        elapsed_time = self.format_time(self.progress_info.get("time_spent", 0))
        self.time_spent_label.config(text=f"Time Spent on Current Image: {elapsed_time}")

        # Update estimated time left if applicable
        estimated_time_left = self.progress_info.get("estimated_time_left")
        self.estimated_time_label.config(text=f"Estimated Time Left: {estimated_time_left}")

        self.frame.after(500, self.update_progress)

    def update_animation(self):
        if self.animation_running:
            self.animation_label.config(text=self.animation_chars[self.animation_index])
            self.animation_index = (self.animation_index + 1) % len(self.animation_chars)
        self.frame.after(250, self.update_animation)

    def update_error_log(self):
        while not self.error_log.empty():
            message = self.error_log.get_nowait()
            self.error_log_text.config(state=tk.NORMAL)
            self.error_log_text.insert(tk.END, f"{message}\n")
            self.error_log_text.see(tk.END)  # Scroll to the end
            self.error_log_text.config(state=tk.DISABLED)
        self.frame.after(1000, self.update_error_log)

class ImageCuttingApp:
    def __init__(self, frame):
        self.frame = frame
        self.input_folder = tk.StringVar(value=f"{project_root}/Upscaled_Images")
        self.output_folder = tk.StringVar(value=f"{folder_path}")
        self.start_from_zero = tk.BooleanVar(value=True)
        self.log_text = None

        from image_cutting.config.text_prompts import text_prompts
        self.text_prompts = text_prompts
        self.selected_tags = {tag: tk.BooleanVar(value=True) for tag in self.text_prompts.keys()}

        self.process_queue = queue.Queue()
        self._setup_gui()

    def _setup_gui(self):
        title_label = tk.Label(self.frame, text="Image Cutting App", font=("InstrumentSans", 16, "bold"))
        title_label.pack(pady=10)

        # Description Label
        description_label = tk.Label(
            self.frame,
            text=(
                "\n ✂️ This section allows you to cut the images based on their content."
                "\n 🔍 We recognise the elements in the images and cut them out."
                "\n 📩 The output will be a series of images, in .JPG format, formatted as the originals in the output folder"
            ),
            wraplength=500,  # Set a maximum width for the text
            justify="left",  # Align the text to the left
        )
        description_label.pack(pady=10)

        # Input Folder Selection
        input_frame = tk.Frame(self.frame)
        input_frame.pack(pady=5, fill="x")
        tk.Label(input_frame, text="Input Folder:", font=("InstrumentSans", 12)).grid(row=0, column=0, sticky="e", padx=5)
        browse_input_button = tk.Button(input_frame, text="Browse", command=self.select_input_folder)
        browse_input_button.grid(row=0, column=1, padx=5)
        # Remove fixed width and use sticky "we"
        self.input_folder_label = tk.Label(input_frame, textvariable=self.input_folder, anchor="w")
        self.input_folder_label.grid(row=0, column=2, sticky="we", padx=5)
        # Make the label column expand
        input_frame.grid_columnconfigure(2, weight=1)

        # Output Folder Selection
        output_frame = tk.Frame(self.frame)
        output_frame.pack(pady=5, fill="x")
        tk.Label(output_frame, text="Output Folder:", font=("InstrumentSans", 12)).grid(row=0, column=0, sticky="e", padx=5)
        browse_output_button = tk.Button(output_frame, text="Browse", command=self.select_output_folder)
        browse_output_button.grid(row=0, column=1, padx=5)
        self.output_folder_label = tk.Label(output_frame, textvariable=self.output_folder, anchor="w")
        self.output_folder_label.grid(row=0, column=2, sticky="we", padx=5)
        output_frame.grid_columnconfigure(2, weight=1)

        # Text input for tags
        tags_frame = tk.LabelFrame(self.frame, text="Select Tags for Cutting, write comma-separated keywords")
        tags_frame.pack(pady=10, fill="x")
        self.custom_tags = tk.StringVar()
        default_tags = ", ".join(self.text_prompts.keys())  # Pre-fill with default tags
        self.custom_tags.set(default_tags)
        tk.Label(tags_frame, text="Tags:").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Entry(tags_frame, textvariable=self.custom_tags, width=50).pack(side=tk.LEFT, padx=5, pady=5)

        # Start From Zero Checkbox
        start_from_zero_checkbox = tk.Checkbutton(
            self.frame, text="Start From Zero", variable=self.start_from_zero
        )
        start_from_zero_checkbox.pack(pady=5)

        # Cutting Button
        cutting_button = tk.Button(self.frame, text="Run Image Cutting", command=self.run_image_cutting)
        cutting_button.pack(pady=10)

        # Log Display
        log_label = tk.Label(self.frame, text="Log:")
        log_label.pack(pady=5)
        self.log_text = tk.Text(self.frame, height=10, state=tk.DISABLED)
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)

        logging.debug(f"Image Cutting frame parent: {self.frame.winfo_parent()}")
        logging.debug(f"Image Cutting frame geometry: {self.frame.winfo_width()}x{self.frame.winfo_height()}")

    def select_input_folder(self):
        folder = fd.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)

    def select_output_folder(self):
        folder = fd.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def run_image_cutting(self):
        """
        Start the image cutting process in a new thread and monitor its progress.
        """
        tag_string = self.custom_tags.get()
        tags = [tag.strip() for tag in tag_string.split(",") if tag.strip()]
        selected_tags = {tag: self.text_prompts.get(tag, 0.25) for tag in tags}

        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        start_from_zero = self.start_from_zero.get()

        if not input_folder or not output_folder:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return
        
        selected_text_prompts = {
            tag: threshold for tag, threshold in self.text_prompts.items()
            if self.selected_tags[tag].get()
        }

        if not selected_text_prompts:
            messagebox.showerror("Error", "Please select at least one tag.")
            return

        self.log_message("Starting image cutting...")
        thread = threading.Thread(
            target=self.process_images_thread,
            args=(input_folder, output_folder, start_from_zero, selected_tags)
        )
        thread.start()
        self.monitor_process()

    def process_images_thread(self, input_folder, output_folder, start_from_zero, selected_tags):
        """
        Thread function to process images with the selected tags.
        """
        try:

            sam_model = initialize_sam_model()
            self.log_message(f"Here is the SAM: {sam_model}")

            process_images(
                input_folder=input_folder,
                output_folder=output_folder,
                sam_model=sam_model,
                start_from_zero=start_from_zero,
                selected_tags=selected_tags,
                log_callback=self.log_message,
            )

            self.process_queue.put("♣️ Queue successfully cut ♣️")
        
        except Exception as e:
            self.process_queue.put(f"error:{str(e)}")

    def monitor_process(self):
        """
        Monitor the process queue for messages and update the UI accordingly.
        """
        try:
            result = self.process_queue.get_nowait()
            if result == "success":
                self.log_message("Image cutting completed successfully!")
                messagebox.showinfo("Success", "Image cutting completed successfully!")
            elif result.startswith("error:"):
                error_message = result.split("error:", 1)[1]
                self.log_message(f"Error during image cutting: {error_message}")
                messagebox.showerror("Error", f"An error occurred: {error_message}")
        except queue.Empty:
            # No messages in the queue yet
            self.frame.after(100, self.monitor_process)

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

class GridCreationApp:
    def __init__(self, frame):
        self.frame = frame
        self._setup_gui()

    def _setup_gui(self):
        # Title Label
        title_label = tk.Label(self.frame, text="Grid Creation App", font=("InstrumentSans", 16, "bold"))
        title_label.pack(pady=10)

        # Description Label
        description_label = tk.Label(
            self.frame,
            text=(
                "\n This section allows you to create a grid layout for your images."
                "\n The script automatically sorts the images based on their nation and category."
                "\n The output will be a series of grids, in .PNG format, to be included in PSD post-processing."
                "\n Specify keywords for nations and categories (comma-separated)."
            ),
            wraplength=400,
            justify="left",
        )
        description_label.pack(pady=10)

        disclaimer_label = tk.Label(
            self.frame,
            text="Note: Please enter a comma-separated list for Nation Keywords and Category Keywords.",
            font=("InstrumentSans", 10, "italic"),
            fg="gray",
        )
        disclaimer_label.pack(pady=5)

        # Input Folder Selection
        input_label = tk.Label(self.frame, text="Input Folder:", font=("InstrumentSans", 12))
        input_label.pack(pady=5)
        input_frame = tk.Frame(self.frame)
        input_frame.pack(pady=5)

        self.input_var = tk.StringVar()
        default_input = f"{project_root}/Upscaled_Images"
        self.input_var.set(default_input)

        def adjust_entry_width(*args):
            # Compute a new width, here we use max(40, number of characters + 2)
            new_width = max(40, len(self.input_var.get()) + 2)
            self.input_entry.config(width=new_width)

        self.input_var.trace_add("write", adjust_entry_width)
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_var, width=40)
        self.input_entry.pack(side="left", padx=5)

        input_browse = tk.Button(input_frame, text="Browse", command=self.browse_input_folder)
        input_browse.pack(side="left", padx=5)

        # Output Folder Selection
        output_label = tk.Label(self.frame, text="Output Folder:", font=("InstrumentSans", 12))
        output_label.pack(pady=5)
        output_frame = tk.Frame(self.frame)
        output_frame.pack(pady=5)
        self.output_entry = tk.Entry(output_frame, width=40)
        self.output_entry.insert(0, f"{project_root}/DB_GRIDS")  # Default path
        self.output_entry.pack(side="left", padx=5)
        output_browse = tk.Button(output_frame, text="Browse", command=self.browse_output_folder)
        output_browse.pack(side="left", padx=5)

        # Nation Pattern Input
        pattern_label = tk.Label(self.frame, text="Nation Keywords (comma-separated):", font=("InstrumentSans", 12))
        pattern_label.pack(pady=5)
        self.pattern_entry = tk.Entry(self.frame, width=50)
        self.pattern_entry.insert(0, "Italian, Greek")
        self.pattern_entry.pack(pady=5)
        self.pattern_entry.bind("<KeyRelease>", self.resize_entry_box)

        # Category Pattern Input
        category_label = tk.Label(self.frame, text="Category Keywords (comma-separated):", font=("InstrumentSans", 12))
        category_label.pack(pady=5)
        self.category_entry = tk.Entry(self.frame, width=50)
        self.category_entry.insert(0, "family,working")
        self.category_entry.pack(pady=5)

        # Grid Size Inputs
        grid_size_label = tk.Label(self.frame, text="Grid Size (Rows x Columns):", font=("InstrumentSans", 12))
        grid_size_label.pack(pady=5)

        # Row input
        row_frame = tk.Frame(self.frame)
        row_frame.pack(pady=5)
        row_label = tk.Label(row_frame, text="Rows:", font=("InstrumentSans", 10))
        row_label.pack(side="left", padx=5)
        self.rows_entry = tk.Entry(row_frame, width=10)
        self.rows_entry.insert(0, "6")  # Default rows
        self.rows_entry.pack(side="left", padx=5)

        # Column input
        col_frame = tk.Frame(self.frame)
        col_frame.pack(pady=5)
        col_label = tk.Label(col_frame, text="Columns:", font=("InstrumentSans", 10))
        col_label.pack(side="left", padx=5)
        self.cols_entry = tk.Entry(col_frame, width=10)
        self.cols_entry.insert(0, "6")  # Default columns
        self.cols_entry.pack(side="left", padx=5)

        # Progress Bar
        progress_label = tk.Label(self.frame, text="Progress:", font=("InstrumentSans", 12))
        progress_label.pack(pady=5)
        self.progress_bar = ttk.Progressbar(self.frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)

        # Run Grid Maker Button
        grid_button = tk.Button(self.frame, text="Run Grid Maker", command=self.run_grid_maker)
        grid_button.pack(pady=10)

    def browse_input_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, folder)

    def browse_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)

    def resize_entry_box(self, event):
        """Dynamically resize the nation keywords entry box based on its content."""
        content_length = len(self.pattern_entry.get())
        new_width = max(10, min(100, content_length))  # Keep the width between 10 and 100
        self.pattern_entry.config(width=new_width)

    def update_progress(self, progress, total):
        """Update the progress bar."""
        self.progress_bar["value"] = (progress / total) * 100
        self.frame.update_idletasks()

    def run_grid_maker(self):
        # Get the input and output directories
        root_dir = self.input_entry.get()
        output_dir = self.output_entry.get()

        # Get the user-defined nation and category patterns
        nation_keywords = self.pattern_entry.get()
        category_keywords = self.category_entry.get()

        # Convert comma-separated keywords into regex patterns
        nation_pattern = "|".join([kw.strip() for kw in nation_keywords.split(",")])
        category_pattern = "|".join([kw.strip() for kw in category_keywords.split(",")])

        # Get grid size from user input
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            grid_size = (rows, cols)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for rows and columns.")
            return

        # Use the create_image_grids function with progress tracking
        try:
            total_steps = 30
            for i in range(total_steps):  # Replace with actual steps in your logic
                self.update_progress(i + 1, total_steps)
            create_image_grids(
                root_dir,
                grid_size=grid_size,
                output_dir=output_dir,
                nation_pattern=nation_pattern,
                category_pattern=category_pattern,
            )
            self.update_progress(total_steps, total_steps)  # Ensure progress is complete
            messagebox.showinfo("Info", "Grid creation completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during grid creation: {e}")

# To initialize the app in main.py
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    root = tk.Tk()
    root.title("EMIF Multi-App GUI")

    # Set the root window geometry and minimum size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width - 100}x{screen_height - 100}+50+50")
    root.minsize(800, 600)
    logging.debug(f"Root window geometry: {root.geometry()}")

    # Create and pack the Notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    root.update_idletasks()
    logging.debug(f"Updated Notebook geometry: {notebook.winfo_width()}x{notebook.winfo_height()}")


    # Tab 1: Image Generation App
    tab1 = tk.Frame(notebook)
    tab1.pack(fill="both", expand=True)
    notebook.add(tab1, text="Image Generation")
    ImageGenerationApp(tab1)

    # Tab 2: Image Cutting App
    tab2 = tk.Frame(notebook)
    tab2.pack(fill="both", expand=True)
    notebook.add(tab2, text="Image Cutting")
    ImageCuttingApp(tab2)

    # Tab 3: Grid Creation App
    tab3 = tk.Frame(notebook)
    tab3.pack(fill="both", expand=True)
    notebook.add(tab3, text="Grid Creation")
    GridCreationApp(tab3)

    notebook.update_idletasks()
    logging.debug(f"Updated Notebook geometry: {notebook.winfo_width()}x{notebook.winfo_height()}")

    # Force geometry update
    root.update_idletasks()
    logging.debug(f"Updated root window geometry: {root.geometry()}")
    logging.debug(f"Updated Notebook geometry: {notebook.winfo_width()}x{notebook.winfo_height()}")

    for i, tab in enumerate(notebook.winfo_children(), start=1):
        logging.debug(f"Tab {i} geometry after add: {tab.winfo_width()}x{tab.winfo_height()}")

    root.mainloop()
