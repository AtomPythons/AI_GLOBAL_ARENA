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
        #self.x = x
        #self.y = y
        self.position = pygame.Vector2(x,y)
        self.radius = radius
        
        #self.vx = 0
        #self.vy = 0
        self.velocity = pygame.Vector2(0,0)
        self.acceleration = pygame.Vector2(0,0)

    def apply_force(self, force):
        self.acceleration += force
        

    def update(self):
        self.velocity += self.acceleration
        self.position += self.velocity

        self.acceleration *=0

        #self.position += self.velocity
        #acceleration_x = 0
        #self.vx += acceleration_x
        #self.x += self.vx
        

        #acceleration_y = 0.118
        #self.vy += acceleration_y
        #self.y += self.vy

        #if self.y + self.radius >= FLOOR_Y:
            #self.y = FLOOR_Y - self.radius
            #self.vy *= -0.80
            
        
        

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            (240, 220, 120),
            (self.position.x, self.position.y), self.radius
        )

class Bone:
    def __init__(self, point_one, point_second, length):
        self.point_a = point_one
        self.point_b = point_second
        self.length = length

    def solve(self):
        delta = self.point_b.position - self.point_a.position
        distance = delta.length()

        if distance == 0:
            return

        difference = (distance - self.length) / distance
        correction = delta * 0.5 * difference

        self.point_a.position += correction
        self.point_b.position -= correction


    def draw(self, screen):
        pygame.draw.line(
            screen,
            (220, 220, 230),
            self.point_a.position,
            self.point_b.position,
            4,
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

    points = [p, p2]


    bone1 = Bone(p, p2, 50)

    bones = [bone1]

    gravity_force = pygame.Vector2(0, 0.1)
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
        for current_point in points:
            current_point.apply_force(gravity_force)

        for current_point in points:
            current_point.update()

        for _ in range(8):
            for bone in bones:
                bone.solve()
                
        draw_world(screen)
        
        for current_point in points:
            current_point.draw(screen)

        
        bone1.draw(screen)
        
        clock.tick(FPS)

        pygame.display.update()


    pygame.quit()

main()
