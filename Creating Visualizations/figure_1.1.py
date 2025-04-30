import tkinter as tk

top_colors = ['#1DFFD6', '#00B6E6', '#B500F4']
bottom_colors = ['#1DFFDA', '#00B6E8', '#B500F7']

root = tk.Tk()
root.title("Color Boxes")
root.geometry("400x300")
root.configure(bg='white')

frame = tk.Frame(root, bg='white')
frame.pack(expand=True, pady=40)

for i, color in enumerate(top_colors):
    box = tk.Frame(frame, bg=color, width=80, height=80)
    box.grid(row=0, column=i, padx=10, pady=5)

for i, color in enumerate(bottom_colors):
    box = tk.Frame(frame, bg=color, width=80, height=80)
    box.grid(row=1, column=i, padx=10, pady=5)

root.mainloop()
