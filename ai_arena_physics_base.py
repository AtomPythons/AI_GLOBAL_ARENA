"""A small Pygame ragdoll sandbox and a starting point for an RL environment."""

import pygame


SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
FLOOR_Y = 450
FPS = 60
LEFT_WALL_X = 0
RIGHT_WALL_X = SCREEN_WIDTH

SUBSTEPS = 3
CONSTRAINT_ITERATIONS = 8
GRAVITY_Y = 900
DAMPING = 0.995
USE_POSTURE_ASSIST = True

BACKGROUND = (31, 36, 45)
FLOOR_COLOR = (52, 56, 65)
POINT_COLOR = (245, 205, 100)
SELECTED_COLOR = (100, 225, 245)
BONE_COLOR = (220, 225, 235)
TEXT_COLOR = (245, 245, 245)


class Point:
    def __init__(self, name, x, y, radius=5):
        self.name = name
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()
        self.radius = radius
        self.is_touching_floor = False

    def apply_force(self, force):
        # Every point has mass 1, so force is also acceleration.
        self.acceleration += force

    def update(self, dt):
        self.velocity += self.acceleration * dt
        self.velocity *= DAMPING ** (dt * FPS)
        self.position += self.velocity * dt
        self.acceleration.update(0, 0)

    def reset_contact_state(self):
        self.is_touching_floor = False

    def solve_world_collision(self):
        if self.position.x - self.radius < LEFT_WALL_X:
            self.position.x = LEFT_WALL_X + self.radius
            self.velocity.x = max(0, -self.velocity.x * 0.2)
        if self.position.x + self.radius > RIGHT_WALL_X:
            self.position.x = RIGHT_WALL_X - self.radius
            self.velocity.x = min(0, -self.velocity.x * 0.2)
        if self.position.y - self.radius < 0:
            self.position.y = self.radius
            self.velocity.y = max(0, -self.velocity.y * 0.2)
        if self.position.y + self.radius >= FLOOR_Y:
            self.position.y = FLOOR_Y - self.radius
            self.is_touching_floor = True
            if self.velocity.y > 0:
                self.velocity.y *= -0.12
            self.velocity.x *= 0.65 if "Foot" in self.name else 0.85

    def draw(self, screen, is_selected=False):
        color = SELECTED_COLOR if is_selected else POINT_COLOR
        pygame.draw.circle(screen, color, self.position, self.radius)
        if is_selected:
            pygame.draw.circle(screen, color, self.position, self.radius + 4, 1)


class Bone:
    def __init__(self, point_a, point_b, length):
        self.point_a = point_a
        self.point_b = point_b
        self.length = length

    def solve(self):
        difference = self.point_b.position - self.point_a.position
        distance = difference.length()
        if distance == 0:
            return
        correction = difference * ((distance - self.length) / distance / 2)
        self.point_a.position += correction
        self.point_b.position -= correction

    def draw(self, screen):
        pygame.draw.line(screen, BONE_COLOR, self.point_a.position,
                         self.point_b.position, 4)


def create_humanoid():
    poses = [
        ("Head", 250, 200), ("Chest", 250, 250), ("Pelvis", 250, 310),
        ("Left Elbow", 220, 265), ("Left Hand", 195, 305),
        ("Right Elbow", 280, 265), ("Right Hand", 305, 305),
        ("Left Knee", 225, 375), ("Left Foot", 220, 445),
        ("Right Knee", 275, 375), ("Right Foot", 280, 445),
    ]
    points = [Point(name, x, y) for name, x, y in poses]
    by_name = {point.name: point for point in points}
    links = [
        ("Head", "Chest", 50), ("Chest", "Pelvis", 60),
        ("Chest", "Left Elbow", 45), ("Left Elbow", "Left Hand", 45),
        ("Chest", "Right Elbow", 45), ("Right Elbow", "Right Hand", 45),
        ("Pelvis", "Left Knee", 65), ("Left Knee", "Left Foot", 70),
        ("Pelvis", "Right Knee", 65), ("Right Knee", "Right Foot", 70),
    ]
    bones = [Bone(by_name[a], by_name[b], length) for a, b, length in links]
    return points, bones


def apply_target_spring(point, target_position, strength, damping):
    target = pygame.Vector2(target_position)
    point.apply_force((target - point.position) * strength
                      - point.velocity * damping)


def apply_posture_assist(points):
    body = {point.name: point for point in points}
    head, chest, pelvis = (body[name] for name in ("Head", "Chest", "Pelvis"))
    apply_target_spring(head, chest.position + (0, -50), 14, 2)
    apply_target_spring(chest, pelvis.position + (0, -60), 12, 2)
    for side in ("Left", "Right"):
        knee = body[side + " Knee"]
        foot = body[side + " Foot"]
        offset = -25 if side == "Left" else 25
        apply_target_spring(knee, pelvis.position + (offset, 65), 10, 1.5)
        apply_target_spring(foot, knee.position + (0, 70), 8, 1.5)
    feet_middle_x = (body["Left Foot"].position.x
                     + body["Right Foot"].position.x) / 2
    apply_target_spring(pelvis, (feet_middle_x, pelvis.position.y), 8, 1)


def get_body_metrics(points):
    body = {point.name: point for point in points}
    head, chest, pelvis = (body[name] for name in ("Head", "Chest", "Pelvis"))
    torso = pelvis.position - chest.position
    torso_angle = abs(pygame.Vector2(0, 1).angle_to(torso)) if torso.length() else 180
    head_height = FLOOR_Y - head.position.y
    pelvis_height = FLOOR_Y - pelvis.position.y
    left_contact = body["Left Foot"].is_touching_floor
    right_contact = body["Right Foot"].is_touching_floor
    center = sum((point.position for point in points), pygame.Vector2()) / len(points)
    score = sum((
        head_height > 160,
        pelvis_height > 90,
        torso_angle < 35,
        left_contact or right_contact,
        head.position.y < chest.position.y < pelvis.position.y,
    ))
    fallen = (head.position.y >= chest.position.y or head_height < 100
              or pelvis_height < 55 or torso_angle > 65)
    return {
        "head_height": head_height,
        "pelvis_height": pelvis_height,
        "torso_angle": torso_angle,
        "left_foot_contact": left_contact,
        "right_foot_contact": right_contact,
        "center_of_mass": center,
        "standing_score": score,
        "is_fallen": fallen,
    }


def calculate_reward(metrics):
    reward = metrics["standing_score"] * 1.0
    reward += max(0, 1 - metrics["torso_angle"] / 65)
    reward += min(1, max(0, metrics["head_height"] / 200))
    reward += min(1, max(0, metrics["pelvis_height"] / 140))
    reward += 0.5 if (metrics["left_foot_contact"]
                      or metrics["right_foot_contact"]) else -1
    reward -= metrics["torso_angle"] / 90
    if metrics["is_fallen"]:
        reward -= 5
    return reward


class EpisodeManager:
    def __init__(self, max_episode_seconds=10):
        self.episode_number = 1
        self.max_episode_seconds = max_episode_seconds
        self.reset()

    def reset(self):
        self.episode_time = 0.0
        self.time_alive = 0.0
        self.fallen_time = 0.0
        self.is_episode_over = False

    def next_episode(self):
        self.episode_number += 1
        self.reset()

    def update(self, dt, is_fallen):
        if self.is_episode_over:
            return
        self.episode_time += dt
        if is_fallen:
            self.fallen_time += dt
        else:
            self.fallen_time = 0.0
            self.time_alive += dt
        self.is_episode_over = (self.fallen_time > 0.3
                                or self.episode_time >= self.max_episode_seconds)


# These hooks can be used by a future learning environment.
def get_observation(points, metrics):
    observation = []
    for point in points:
        observation.extend((point.position.x / SCREEN_WIDTH,
                            point.position.y / FLOOR_Y,
                            point.velocity.x / 500,
                            point.velocity.y / 500))
    observation.extend((metrics["torso_angle"] / 180,
                        float(metrics["left_foot_contact"]),
                        float(metrics["right_foot_contact"])))
    return observation


def get_action_space():
    return ["do_nothing", "chest_up", "chest_left", "chest_right",
            "pelvis_up", "pelvis_left", "pelvis_right",
            "left_foot_left", "left_foot_right",
            "right_foot_left", "right_foot_right"]


def apply_discrete_action(points, action_index):
    actions = get_action_space()
    if not 0 <= action_index < len(actions):
        raise ValueError("Invalid action index")
    action = actions[action_index]
    if action == "do_nothing":
        return
    names = {"chest": "Chest", "pelvis": "Pelvis",
             "left_foot": "Left Foot", "right_foot": "Right Foot"}
    directions = {"up": (0, -1200), "left": (-1200, 0),
                  "right": (1200, 0)}
    target, direction = action.rsplit("_", 1)
    next(point for point in points if point.name == names[target]).apply_force(
        pygame.Vector2(directions[direction]))


def apply_manual_force(point, keys, mouse_position=None):
    direction = pygame.Vector2(
        int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
        int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP]),
    )
    if direction.length_squared():
        point.apply_force(direction.normalize() * 1800)
    if mouse_position is not None:
        # A spring lets the mouse pull the body without teleporting a point.
        apply_target_spring(point, mouse_position, 120, 15)


def find_point_at_mouse(points, mouse_position):
    mouse = pygame.Vector2(mouse_position)
    closest = min(points, key=lambda point:
                  point.position.distance_squared_to(mouse))
    grab_radius = max(closest.radius + 10, 16)
    if closest.position.distance_squared_to(mouse) <= grab_radius ** 2:
        return closest
    return None


def draw_scene(screen, font, small_font, points, bones, selected,
               show_labels, metrics, reward, episode, posture_assist):
    screen.fill(BACKGROUND)
    pygame.draw.rect(screen, FLOOR_COLOR,
                     (0, FLOOR_Y, SCREEN_WIDTH, SCREEN_HEIGHT - FLOOR_Y))
    pygame.draw.line(screen, BONE_COLOR, (0, FLOOR_Y),
                     (SCREEN_WIDTH, FLOOR_Y), 2)
    for bone in bones:
        bone.draw(screen)
    for point in points:
        point.draw(screen, point is selected)
        if show_labels:
            label = small_font.render(point.name, True, TEXT_COLOR)
            screen.blit(label, point.position + (8, -12))

    lines = [
        f"Episode: {episode.episode_number}  Time: {episode.episode_time:.1f}s",
        f"Time alive: {episode.time_alive:.1f}s",
        f"Status: {'Fallen' if metrics['is_fallen'] else 'Standing'}"
        + (" (episode over)" if episode.is_episode_over else ""),
        f"Standing score: {metrics['standing_score']}/5  Reward: {reward:.2f}",
        f"Torso angle: {metrics['torso_angle']:.1f} deg",
        f"Head height: {metrics['head_height']:.1f}",
        f"Pelvis height: {metrics['pelvis_height']:.1f}",
        f"Selected: {selected.name}  Assist: {'on' if posture_assist else 'off'}",
    ]
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, TEXT_COLOR), (8, 7 + index * 18))
    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Arena - Physics Base")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    small_font = pygame.font.SysFont(None, 16)
    points, bones = create_humanoid()
    selected = points[0]
    episode = EpisodeManager()
    show_labels = False
    posture_assist = USE_POSTURE_ASSIST
    dragging = False
    number_keys = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                   pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                   pygame.K_9, pygame.K_0, pygame.K_MINUS)
    running = True

    while running:
        dt = min(clock.tick(FPS) / 1000, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in number_keys:
                    selected = points[number_keys.index(event.key)]
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_p:
                    posture_assist = not posture_assist
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        episode.next_episode()
                    else:
                        episode.reset()
                    points, bones = create_humanoid()
                    selected = points[0]
                    dragging = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_point = find_point_at_mouse(points, event.pos)
                if clicked_point is not None:
                    selected = clicked_point
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

        sub_dt = dt / SUBSTEPS
        keys = pygame.key.get_pressed()
        for _ in range(SUBSTEPS):
            for point in points:
                point.reset_contact_state()
                point.apply_force(pygame.Vector2(0, GRAVITY_Y))
            if posture_assist:
                apply_posture_assist(points)
            apply_manual_force(selected, keys,
                               pygame.mouse.get_pos() if dragging else None)
            previous = [point.position.copy() for point in points]
            for point in points:
                point.update(sub_dt)
            for _ in range(CONSTRAINT_ITERATIONS):
                for bone in bones:
                    bone.solve()
                for point in points:
                    point.solve_world_collision()
            # Account for movement caused by the distance constraints.
            for point, old_position in zip(points, previous):
                point.velocity = (point.position - old_position) / sub_dt
                point.solve_world_collision()
        metrics = get_body_metrics(points)
        episode.update(dt, metrics["is_fallen"])
        reward = calculate_reward(metrics)
        draw_scene(screen, font, small_font, points, bones, selected,
                   show_labels, metrics, reward, episode, posture_assist)

    pygame.quit()


if __name__ == "__main__":
    main()
