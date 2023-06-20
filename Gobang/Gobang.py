# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter.messagebox import showinfo


class Gobang(object):

    def __init__(self, master, board_size=15):
        self.master = master
        self.master.title('Gobang Game')
        self.board_size = board_size
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1
        self.game_over = False
        self.create_gui()

    def reset(self):
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        self.game_over = False
        return self.board, self.game_over

    def get_xy(self, action):
        return action // len(self.board), action % len(self.board[0])

    def step(self, action, display=False):
        x, y = self.get_xy(action=action)

        if display:
            self.self_play_display(position=(x, y))

        reward = self.cmd_play((x, y))
        if reward:
            return self.board, 1.0, reward
        return self.board, -1.0, reward

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

        if self.board[x][y] != 0:
            return None

        self.board[x][y] = self.current_player

        if self.check_win():
            self.game_over = True
            print(f'Player {self.current_player} wins!')

        self.current_player = 3 - self.current_player

        return self.game_over

    def self_play_display(self, position):
        y, x = position[0], position[1]

        if self.current_player == 1:
            color = 'black'
        else:
            color = 'white'

        self.canvas.create_oval(x * 40 + 5, y * 40 + 5, x * 40 + 35, y * 40 + 35, fill=color)
        self.canvas.update()

    def place_piece(self, event):
        x, y = event.x // 40, event.y // 40

        if self.current_player == 1:
            color = 'black'
        else:
            color = 'white'
        self.canvas.create_oval(x * 40 + 5, y * 40 + 5, x * 40 + 35, y * 40 + 35, fill=color)
        self.canvas.update()
        self.modify_board(position=(y, x))

        if self.game_over:
            if (3 - self.current_player) % 2 == 1:
                showinfo(title='Hi', message='Player Black Wins!')
            elif (3 - self.current_player) % 2 == 0:
                showinfo(title='Hi', message='Player White Wins!')
            exit(0)

    def check_win(self):
        rows = len(self.board)
        cols = len(self.board[0])

        for row in range(rows):
            for col in range(cols - 4):
                if all(self.board[row][col + i] == self.current_player for i in range(5)):
                    return True
        for col in range(cols):
            for row in range(rows - 4):
                if all(self.board[row + i][col] == self.current_player for i in range(5)):
                    return True

        for row in range(rows - 4):
            for col in range(cols - 4):
                if all(self.board[row + i][col + i] == self.current_player for i in range(5)):
                    return True

        for row in range(rows - 4):
            for col in range(4, cols):
                if all(self.board[row + i][col - i] == self.current_player for i in range(5)):
                    return True

        return False


if __name__ == '__main__':
    root = tk.Tk()
    game = Gobang(root)
    root.mainloop()
