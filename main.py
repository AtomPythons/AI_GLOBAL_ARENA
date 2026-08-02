import pygame
import time

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

FPS = 60

BACKGROUND_COLOR = (255, 0, 0)

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
            
                
        screen.fill((BACKGROUND_COLOR))
        clock.tick(FPS)

        pygame.display.update()


    pygame.quit()

main()
