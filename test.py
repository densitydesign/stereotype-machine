import tkinter

root = tkinter.Tk()  # Create a Tk instance
print(root.tk.call('info', 'patchlevel'))  # Retrieve Tcl/Tk version
root.destroy()