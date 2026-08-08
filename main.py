import pygame
import time

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

FPS = 60

BACKGROUND_COLOR = (25, 30, 30)
GREEN = (0, 255, 0)
DARK_GREY = (60, 60, 70)

LINE_COLOR = (200, 200, 200)

class Point:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

        self.velocity_y = 1

    def update(self):
        self.y += self.velocity_y
        

    def draw(self,screen):
        pygame.draw.circle(
            screen,
            (240, 220, 120),
            (self.x, self.y), self.radius
        )
            

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

    p = Point(100, 100, 8)


    running = True
    while running:
        all_events = pygame.event.get()
        for event in all_events:
            if event.type == pygame.QUIT:
                running = False
        p.update()    
                
        draw_world(screen)
        p.draw(screen)
        
        clock.tick(FPS)

        pygame.display.update()


    pygame.quit()

main()
