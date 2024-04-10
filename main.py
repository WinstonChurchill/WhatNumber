import pygame
import sys
import numpy as np

from settings import *
from neronka import *

if __name__ != '__main__':
    sys.exit()

maping = np.zeros((28, 28))

pygame.init()

screen = pygame.display.set_mode(screen_main)

def otrisovka(screen, mapping):
    for x in range(COL_YACHE + 1):
        pygame.draw.line(screen, COLOR_2, (x * cof + TOLHINA // 2, 0), (x * cof + TOLHINA // 2, cof * 28), TOLHINA)
    for y in range(COL_YACHE + 1):
        pygame.draw.line(screen, COLOR_2, (0, y * cof + TOLHINA // 2), (cof * 28, y * cof + TOLHINA // 2), TOLHINA)
    for x in range(COL_YACHE):
        for y in range(COL_YACHE):
            if maping[y][x] == 1:
                pygame.draw.rect(screen, COLOR_1, (x * cof + TOLHINA, y * cof + TOLHINA, cof - TOLHINA, cof - TOLHINA), 0)

shift = pygame.font.Font('text.ttf', cof_text)
def otrisovka_gui(screen, verot):
    num = -1
    for i in verot:
        num += 1
        text1 = shift.render(f'{num}: {str(i[0])[:11]}', True, (COLOR_1))
        screen.blit(text1, (w + cof_text, -1 + num * cof_text + 2))
    text1 = shift.render(f'Это:{veroit.argmax()}', True, (COLOR_1))
    screen.blit(text1, (w + cof_text, -1 + len(veroit) * cof_text + 2))

def iterachion(maping):
    global veroit
    mapping_array_prav = np.reshape(maping, (-1, 1))

    print(mapping_array_prav)

    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ mapping_array_prav
    hidden = 1 / (1 + np.exp(-hidden_raw))

    print(hidden)

    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    print(output)

    veroit = output

def opred_kletk_and_edit(pos, left_mouse_button, prochet, maping):
    x = pos[0] // cof
    y = pos[1] // cof
    if left_mouse_button and x < COL_YACHE and x > -1 and y < COL_YACHE and y > -1:
        if maping[y][x] == 0:
            maping[y][x] = 1
        if y < len(maping) - 1:
            maping[y + 1][x] = 1
        if y > 0:
            maping[y - 1][x] = 1
        if x < len(maping) - 1:
            maping[y][x + 1] = 1
        if x > 0:
            maping[y][x - 1] = 1
        if prochet:
                iterachion(maping)
    elif left_mouse_button == 0 and y < COL_YACHE and y > -1 and x < COL_YACHE and x > -1: 
        if maping[y][x] == 1:
            if prochet:
                iterachion(maping)
            maping[y][x] = 0

clock = pygame.time.Clock()
pygame.display.set_icon(pygame.image.load("icon.bmp"))

#flags
prochet = 1
mouse_button = 0
mouse_button_left = 0
mouse_button_right = 0

cash = load_neronka('vesa_ner.json')
weights_input_to_hidden = cash['weights_input_to_hidden']
weights_hidden_to_output = cash['weights_hidden_to_output']
bias_input_to_hidden = cash['bias_input_to_hidden']
bias_hidden_to_output = cash['bias_hidden_to_output']

veroit = weights_hidden_to_output

# weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
# weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
# bias_input_to_hidden = np.zeros((20, 1))
# bias_hidden_to_output = np.zeros((10, 1))

# save_neronka('vesa_ner', {'weights_input_to_hidden': weights_input_to_hidden.tolist(), 'weights_hidden_to_output': weights_hidden_to_output.tolist(),
# 'bias_input_to_hidden': bias_input_to_hidden.tolist(), 'bias_hidden_to_output': bias_hidden_to_output.tolist()})

while True:
    pygame.display.set_caption(f'{title} FPS:{int(clock.get_fps())}')
    screen.fill(COLOR_0)
    otrisovka(screen, maping)
    otrisovka_gui(screen, veroit)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_button = 1
            if pygame.mouse.get_pressed()[0] == 1:
                mouse_button_left = 1
            elif pygame.mouse.get_pressed()[2] == 1:
                mouse_button_right = 1
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_button_left = 0
            mouse_button_right = 0
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                maping = np.zeros((28, 28))
                if prochet:
                    iterachion(maping)
            elif event.key == pygame.K_e:
                prochet = not(prochet)
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    if mouse_button:
        if mouse_button_left:
            opred_kletk_and_edit(pygame.mouse.get_pos(), 1, prochet, maping)
        elif mouse_button_right:
            opred_kletk_and_edit(pygame.mouse.get_pos(), 0, prochet, maping)

    pygame.display.flip()
    clock.tick(fps)
