import tkinter as tk
from screens import show_main_screen
import state

def main():
    root = tk.Tk()
    state.root = root
    root.title("Oil Classifier")
    root.geometry("600x400")
    root.minsize(600, 400)
    show_main_screen()
    root.mainloop()

if __name__ == "__main__":
    main()