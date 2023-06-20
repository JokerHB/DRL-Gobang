# -*- coding: utf-8 -*-

import tkinter as tk
from Gobang.Gobang import Gobang

if __name__ == '__main__':
    root = tk.Tk()
    game = Gobang(root)
    root.mainloop()
