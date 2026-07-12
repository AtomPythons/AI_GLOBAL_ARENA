import pygame
import time

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

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
            
                
        screen.fill((255,0,0))
        clock.tick(60)

        pygame.display.update()


    pygame.quit()

main()
