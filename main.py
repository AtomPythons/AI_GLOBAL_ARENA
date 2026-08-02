import pygame
import time

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

FPS = 60

BACKGROUND_COLOR = (25, 30, 30)
GREEN = (0, 255, 0)
DARK_GREY = (60, 60, 70)

LINE_COLOR = (200, 200, 200)


def draw_world(screen):
    screen.fill(BACKGROUND_COLOR)

    pygame.draw.rect(screen, DARK_GREY, pygame.Rect(0, 0, 500, 450))

    pygame.draw.line(screen, LINE_COLOR, (0, 450), (500, 450), 2)

def main():
    # SOME CODE FROM DEVELOPER(IGOR)
    #ok
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()


    running = True
    while running:
        all_events = pygame.event.get()
        for event in all_events:
            if event.type == pygame.QUIT:
                running = False
            
                
        draw_world(screen)
        
        clock.tick(FPS)

        pygame.display.update()


    pygame.quit()

main()
