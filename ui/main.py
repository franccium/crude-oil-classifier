import tkinter as tk
from ui.screens import show_main_screen
import ui.state as state

def main():
    root = tk.Tk()
    state.root = root
    root.title("Oil Classifier")
    root.geometry("800x400")
    root.minsize(800, 500)
    root.maxsize(800, 400)
    show_main_screen()
    root.mainloop()

if __name__ == "__main__":
    main()