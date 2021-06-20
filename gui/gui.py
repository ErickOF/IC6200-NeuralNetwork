import tkinter as tk
import random


# Defining some constants
TITLE = 'R&M'
# Dimensions
HEIGHT, WIDTH = 500, 570
# Background color
BG_COLOR = 'violet'
# Button color
BG_BTN = 'purple'
# Button text color
BTN_TEXT_COLOR = 'white'
# Text font
TEXT_FONT = 'Execute'
# Text size
TEXT_SIZE = 30
# Button text size
BTN_TEXT_SIZE = 12
# Board
grid = []

#Board Color
grid_color = []
# Gameboard dimensions 8x8
N, M = 10, 10

# Create a main window
window = tk.Tk()

# Get screen dimensions
SCREEN_WIDTH = window.winfo_screenwidth()
SCREEN_HEIGHT = window.winfo_screenheight()

# Get new window position
WINDOW_POS_X = int((SCREEN_WIDTH/2) - (WIDTH/2))
WINDOW_POS_Y = int((SCREEN_HEIGHT/2) - (HEIGHT/2))

# Set title
window.title(TITLE)
# Set icon
#window.iconbitmap('src/icons/py.ico')
# Set geometry
# WIDTHxHIGHT+X+Y
window.geometry(f'{WIDTH}x{HEIGHT}+{WINDOW_POS_X}+{WINDOW_POS_Y}')
# Set if it's resizable
window.resizable(width=False, height=False)
# Set background
window.config(bg=BG_COLOR)


def on_click(i,j,event):
        print(grid[i][j].cget('bg'))
        color = grid[i][j].cget('bg')
        if color == 'black':
            color = 'white'
        else:
            color = 'black'
        grid[i][j].config(bg=color)
        # Create grid
        grid[i][j].grid(row=i, column=j)
        grid_color[(i*10)+j] = color

# Init gameboard
grid = [[tk.Label(window, width=7, height=2, borderwidth=3, relief='solid') for j in range(M)] for i in range(N)]

grid_color = ['white' for j in range(M*N)]

#Bind color change to labels
for i, row in enumerate(grid):
    for j, column in enumerate(row):
        grid[i][j].bind('<Button-1>', lambda e, i=i, j=j: on_click(i,j,e))

def drawGrid(grid):
    """This function displays gameboard.
    """
    # Color to start drawing
    color = 'white'

    for i in range(N):
        for j in range(M):
            # Change label bg
            grid[i][j].config(bg=color)
            # Create grid
            grid[i][j].grid(row=i, column=j)
            # Change color
            color = 'white' 
        # Change color
        color = 'white' 

# Draw grid
drawGrid(grid)

def run():
    print(grid_color)

#Button
btnRun = tk.Button(window, text='Execute', bg=BG_BTN,
                          fg=BTN_TEXT_COLOR, font=(TEXT_FONT, BTN_TEXT_SIZE),
                          activebackground=BTN_TEXT_COLOR,
                          activeforeground=BG_BTN,
                          command=run) # Add command
btnRun.place(x=500, y=385)


# Call main loop
window.mainloop()