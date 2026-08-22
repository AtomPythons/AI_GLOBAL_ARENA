import pygame
import time

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

FLOOR_Y = 450

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
        
        self.vx = 0
        self.vy = 0

    def update(self):

        acceleration_x = 0
        self.vx += acceleration_x
        self.x += self.vx

        acceleration_y = 0.118
        self.vy += acceleration_y
        self.y += self.vy

        if self.y + self.radius >= FLOOR_Y:
            self.y = FLOOR_Y - self.radius
            self.vy *= -0.80
            
        
        

    def draw(self,screen):
        pygame.draw.circle(
            screen,
            (240, 220, 120),
            (self.x, self.y), self.radius
        )

            

def draw_world(screen):
    screen.fill(BACKGROUND_COLOR)

    pygame.draw.rect(screen, DARK_GREY, pygame.Rect(0, 0, 500, FLOOR_Y))

    pygame.draw.line(screen, LINE_COLOR, (0, FLOOR_Y), (500, FLOOR_Y), 2)

def main():
    # SOME CODE FROM DEVELOPER(IGOR)
    #ok
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    p = Point(100, 100, 8)
    p2 = Point(400, 100, 8)


    running = True
    while running:
        all_events = pygame.event.get()
        for event in all_events:
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] == True:
            p.vy -= 0.1
        if keys[pygame.K_DOWN] == True:
            p.vy += 0.1
        if keys[pygame.K_LEFT] == True:
            p.vx -= 0.3
        if keys[pygame.K_RIGHT] == True:
            p.vx += 0.3
        if keys[pygame.K_w] == True:
            p2.vy -= 0.1
        if keys[pygame.K_s] == True:
            p2.vy += 0.1
        if keys[pygame.K_a] == True:
            p2.vx -= 0.3
        if keys[pygame.K_d] == True:
            p2.vx += 0.3
        
        p.update()
        p2.update()
                
        draw_world(screen)
        p.draw(screen)
        p2.draw(screen)
        
        clock.tick(FPS)

        pygame.display.update()


    pygame.quit()

main()
