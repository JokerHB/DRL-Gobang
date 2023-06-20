# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter.messagebox import showinfo


class Gobang(object):

    def __init__(self, master):
        self.master = master
        self.master.title('Gobang Game')
        self.board = [[0 for _ in range(15)] for _ in range(15)]
        self.current_player = 1
        self.game_over = False
        self.create_gui()

    def create_gui(self):
        self.canvas = tk.Canvas(self.master, width=600, height=600, bg='bisque')
        screenWidth = self.master.winfo_screenwidth()
        screenHeight = self.master.winfo_screenheight()

        centerX = int((screenWidth - 600) / 2)
        centerY = int((screenHeight - 600) / 2)

        self.master.geometry('600x600+{}+{}'.format(centerX, centerY))

        self.canvas.pack()

        for i in range(16):
            self.canvas.create_line(40 * i, 0, 40 * i, 600)
            self.canvas.create_line(0, 40 * i, 600, 40 * i)

        self.canvas.bind('<Button-1>', self.place_piece)

    def cmd_play(self, position):
        return self.modify_board(position=position)

    def modify_board(self, position):
        x, y = position[0], position[1]

        if self.board[y][x] != 0:
            return None

        self.board[y][x] = self.current_player

        if self.check_win(x, y):
            self.game_over = True
            print(f'Player {self.current_player} wins!')

        self.current_player = 3 - self.current_player

        return self.game_over

    def place_piece(self, event):
        x, y = event.x // 40, event.y // 40

        if self.current_player == 1:
            color = 'black'
        else:
            color = 'white'
        self.canvas.create_oval(x * 40 + 5, y * 40 + 5, x * 40 + 35, y * 40 + 35, fill=color)
        self.canvas.update()
        self.modify_board(position=(x, y))

        if self.game_over:
            if (3 - self.current_player) % 2 == 1:
                showinfo(title='Hi', message='Player Black Wins!')
            elif (3 - self.current_player) % 2 == 0:
                showinfo(title='Hi', message='Player White Wins!')
            exit(0)

    def check_win(self, x, y):
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for direction in [(-1, -1), (1, 1)]:
                for step in range(1, 5):
                    new_x = x + direction[0] * dx * step
                    new_y = y + direction[1] * dy * step

                    if 0 <= new_x < 15 and 0 <= new_y < 15 and self.board[new_y][new_x] == self.current_player:
                        count += 1
                    else:
                        break

            if count >= 5:
                return True

        return False


if __name__ == '__main__':
    root = tk.Tk()
    game = Gobang(root)
    root.mainloop()
