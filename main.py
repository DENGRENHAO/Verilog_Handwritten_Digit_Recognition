from utils import *
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drawing program")

def init_grid(rows, cols, color):
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            grid[i].append(color)

    return grid

def draw_grid(window, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            if pixel == BLACK:
                pygame.draw.circle(window, pixel, (j * PIXEL_SIZE, i * PIXEL_SIZE + TOOLBAR_HEIGHT), 10)
    
    if DRAW_GRID_LINES:
        for i in range(ROWS + 1):
            pygame.draw.line(window, BLACK, (0, i * PIXEL_SIZE + TOOLBAR_HEIGHT), (WIDTH, i * PIXEL_SIZE + TOOLBAR_HEIGHT))
        for i in range(COLS + 1):
            pygame.draw.line(window, BLACK, (i * PIXEL_SIZE, TOOLBAR_HEIGHT), (i * PIXEL_SIZE, HEIGHT))

def draw(window, grid):
    window.fill(BG_COLOR)
    draw_grid(window, grid)
    pygame.display.update()

def get_rol_col_from_pos(pos):
    x, y = pos
    row = (y - TOOLBAR_HEIGHT) // PIXEL_SIZE
    col = x // PIXEL_SIZE

    if row < 0:
        raise IndexError

    return row, col

def display_text(window, text, fontsize):
    font = pygame.font.SysFont("comicsans", fontsize)
    text = font.render(text, True, BLACK)
    textRect = text.get_rect()
    textRect.center = (WIDTH // 2, TOOLBAR_HEIGHT // 2)
    window.blit(text, textRect)
    pygame.display.update()

def draw_frame(window):
    pygame.draw.line(window, BLACK, (0, TOOLBAR_HEIGHT), (WIDTH, TOOLBAR_HEIGHT))
    pygame.display.update()


def output_image_file(img, filename):
    orig_img = img * 255  # type(image) = numpy.ndarray
    orig_img = orig_img.astype('uint8')
    # output as hex file (%x means hexadecimal)
    orig_img.tofile(os.path.join("./data", filename), sep=' ', format='%x')

def predict(window):
    cropped_region = (0, TOOLBAR_HEIGHT, WIDTH, HEIGHT - TOOLBAR_HEIGHT)
    cropped_window = window.subsurface(cropped_region)

    img_file = "./data/img.png"
    pygame.image.save(cropped_window, img_file)
    
    orig_image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(orig_image, (28, 28), interpolation=cv2.INTER_AREA)
    img_resized = cv2.bitwise_not(img_resized)
    img_resized = img_resized * 0.9

    try:
        # plot_img(img_resized)
        output_image_file(img_resized, 'hex_img.hex')
        os.system("iverilog -o ./verilog/nn_recognize ./verilog/nn_recognize.v")
        stream = os.popen("vvp ./verilog/nn_recognize")
        output = stream.read()
        return output
    except:
        return -1

def plot_img(ndarr):
    img = ndarr.copy()
    plt.imshow(img, cmap='gray')
    plt.show()

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BG_COLOR)
left_click_action = "DOWN"
stop_time = pygame.time.get_ticks()
stopped = False

while run:
    clock.tick(FPS)
    
    draw_frame(WINDOW)

    if stopped and (pygame.time.get_ticks() - stop_time) > 1500:
        left_click_action = "DOWN"
        grid = init_grid(ROWS, COLS, BG_COLOR)
        draw(WINDOW, grid)
        stopped = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if stopped == False:
            if pygame.mouse.get_pressed()[0]:
                left_click_action = "DOWN"
                pos = pygame.mouse.get_pos()
                try:
                    row, col = get_rol_col_from_pos(pos)
                    grid[row][col] = BLACK
                except IndexError:
                    pass

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    left_click_action = "UP"
                    left_click_up_time = pygame.time.get_ticks()

            draw(WINDOW, grid)

    if stopped == False and left_click_action == "UP" and (pygame.time.get_ticks() - left_click_up_time) > 1000:
        number = int(predict(WINDOW))
        fontsize = 28
        if number != -1:
            text = f"This number is: {number}"
        else:
            fontsize = 14
            text = "Can't recognize. Try another one."
        display_text(WINDOW, text, fontsize)
        stop_time = pygame.time.get_ticks()
        stopped = True

pygame.quit()