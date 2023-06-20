# -*- coding: utf-8 -*-

import tkinter as tk
from DRL.AC import AC
from Gobang.Gobang import Gobang

if __name__ == '__main__':
    root = tk.Tk()
    game = Gobang(root)
    # root.mainloop()

    ac = AC(input_size=15 * 15, hidden_size=512, output_size=15 * 15)
    ac.train_model(env=game)
