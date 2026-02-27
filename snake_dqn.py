import os
import random
import math
import time
from array import array
from collections import deque
from dataclasses import dataclass

import pygame
import torch
import torch.nn as nn
import torch.optim as optim


# Optional headless mode for notebook/Colab training.
HEADLESS = os.environ.get("HEADLESS", "0").strip() == "1"
if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Game constants
WINDOW_SIZE = 900
GRID_SIZE = 20
BLOCK_SIZE = WINDOW_SIZE // GRID_SIZE
FPS_LEVELS = {1: 6, 2: 10, 3: 14}
FPS_AI = 15

# Colors (RGB) - pastel pink dreamy theme
BG_TOP = (255, 215, 230)
BG_BOTTOM = (255, 190, 220)
GRID_COLOR = (255, 235, 245)
SNAKE_BODY = (120, 200, 170)
SNAKE_HEAD = (70, 160, 130)
FOOD_NORMAL = (255, 90, 140)
FOOD_BONUS = (255, 130, 160)
FOOD_POISON = (160, 90, 200)
TEXT_COLOR = (80, 40, 60)
OVERLAY_COLOR = (255, 255, 255)

FOOD_TYPES = [
    ("normal", FOOD_NORMAL, 10, 1, 0.7),
    ("bonus", FOOD_BONUS, 20, 2, 0.2),
    ("poison", FOOD_POISON, -8, -1, 0.1),
]

TRAIL_LENGTH = 12
COMBO_WINDOW = 15
COMBO_BONUS = 2

MODEL_PATH = "model.pth"
CURVE_PATH = "training_curve.png"

ACTION_NAMES = {
    0: "Left",
    1: "Straight",
    2: "Right",
}


directions = [
    pygame.Vector2(0, -1),  # up
    pygame.Vector2(1, 0),   # right
    pygame.Vector2(0, 1),   # down
    pygame.Vector2(-1, 0),  # left
]


def clamp(val, low, high):
    return max(low, min(high, val))


def load_best_score():
    if not os.path.exists("best_score.txt"):
        return 0
    try:
        with open("best_score.txt", "r", encoding="utf-8") as handle:
            return int(handle.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def save_best_score(score):
    try:
        with open("best_score.txt", "w", encoding="utf-8") as handle:
            handle.write(str(score))
    except OSError:
        return


@dataclass
class GameState:
    score: int = 0
    episode: int = 0
    epsilon: float = 0.0
    avg_score: float = 0.0
    mode: str = "manual"
    manual_started: bool = False
    last_action: int = 1
    last_reward: float = 0.0
    last_state: list | None = None
    best_score: int = 0
    combo: int = 0
    manual_speed: int = 1
    food_type: str = "normal"


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden3)
        self.linear4 = nn.Linear(hidden3, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name, device):
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name, map_location=device))


class DQNAgent:
    def __init__(self, device):
        self.device = device
        self.epsilon = 0.8
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        self.model = LinearQNet(11, 256, 128, 64, 3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, greedy=False):
        if (not greedy) and (random.random() < self.epsilon):
            return random.randint(0, 2)
        state0 = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model(state0)
            return torch.argmax(prediction).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        pred = self.model(states).gather(1, actions)
        with torch.no_grad():
            target = rewards + (1 - dones) * self.gamma * self.model(next_states).max(1, keepdim=True)[0]

        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        self.model.save(MODEL_PATH)

    def load(self):
        self.model.load(MODEL_PATH, self.device)


class SnakeGameAI:
    def __init__(self):
        pygame.init()
        self._init_audio()
        pygame.display.set_caption("AI Snake - DQN")
        self.display = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        # Use default font to avoid platform sysfont registry issues.
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 56)
        self.last_game_over_shot_time = 0.0
        self.start_time = time.time()
        self.background = self._build_background()
        self.last_event = None

        self.reset()

    def reset(self):
        self.direction = pygame.Vector2(1, 0)
        head = pygame.Vector2(GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake = [
            head,
            head - pygame.Vector2(1, 0),
            head - pygame.Vector2(2, 0),
        ]
        self.trail = deque(maxlen=TRAIL_LENGTH)
        self.steps_since_food = 0
        self.combo = 0
        self.score = 0
        self.frame_count = 0
        self.place_food()
        self.game_over = False
        self.game_over_time = 0
        self.game_over_sound_played = False
        self.start_time = time.time()

    def place_food(self):
        roll = random.random()
        cumulative = 0.0
        chosen = FOOD_TYPES[0]
        for item in FOOD_TYPES:
            cumulative += item[4]
            if roll <= cumulative:
                chosen = item
                break
        self.food_type = chosen[0]
        self.food_color = chosen[1]
        self.food_reward = chosen[2]
        self.food_score = chosen[3]
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            food = pygame.Vector2(x, y)
            if food not in self.snake:
                self.food = food
                return

    def is_collision(self, point):
        if point.x < 0 or point.x >= GRID_SIZE or point.y < 0 or point.y >= GRID_SIZE:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def play_step(self, action):
        self.frame_count += 1
        self._move(action)
        self.snake.insert(0, self.head)
        self.trail.appendleft(self.head)
        self.steps_since_food += 1
        self.last_event = None

        reward = 0.1
        done = False

        # Optional distance-based shaping
        prev_dist = self._distance_to_food(self.snake[1])
        new_dist = self._distance_to_food(self.head)
        if new_dist > prev_dist:
            reward -= 0.5

        if self.is_collision(self.head):
            reward = -10
            done = True
            self.last_event = "death"
            return self.get_state(), reward, done, self.score

        if self.head == self.food:
            if self.steps_since_food <= COMBO_WINDOW:
                self.combo += 1
            else:
                self.combo = 1

            reward = self.food_reward + max(0, self.combo - 1) * COMBO_BONUS
            self.score = max(0, self.score + self.food_score)
            self.steps_since_food = 0
            if self.food_type == "poison":
                self.last_event = "poison"
                self.snake.pop()
            else:
                self.last_event = "bonus" if self.food_type == "bonus" else "eat"
            self.place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, self.score

    def _distance_to_food(self, point):
        return abs(point.x - self.food.x) + abs(point.y - self.food.y)

    def _move(self, action):
        # action: 0 left, 1 straight, 2 right
        idx = directions.index(self.direction)
        if action == 0:
            idx = (idx - 1) % 4
        elif action == 2:
            idx = (idx + 1) % 4
        self.direction = directions[idx]
        self.head = self.snake[0] + self.direction

    def get_state(self):
        head = self.snake[0]
        point_straight = head + self.direction
        point_right = head + directions[(directions.index(self.direction) + 1) % 4]
        point_left = head + directions[(directions.index(self.direction) - 1) % 4]

        danger_straight = 1 if self.is_collision(point_straight) else 0
        danger_right = 1 if self.is_collision(point_right) else 0
        danger_left = 1 if self.is_collision(point_left) else 0

        food_left = 1 if self.food.x < head.x else 0
        food_right = 1 if self.food.x > head.x else 0
        food_up = 1 if self.food.y < head.y else 0
        food_down = 1 if self.food.y > head.y else 0

        dir_left = 1 if self.direction == directions[3] else 0
        dir_right = 1 if self.direction == directions[1] else 0
        dir_up = 1 if self.direction == directions[0] else 0
        dir_down = 1 if self.direction == directions[2] else 0

        return [
            danger_straight,
            danger_right,
            danger_left,
            food_left,
            food_right,
            food_up,
            food_down,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
        ]

    def draw(self, game_state):
        self._draw_background()

        # Mode badge
        mode_label = f"MODE: {game_state.mode.upper()}"
        badge_text = self.font.render(mode_label, True, TEXT_COLOR)
        badge_padding = 8
        badge_rect = pygame.Rect(
            (WINDOW_SIZE - badge_text.get_width()) // 2 - badge_padding,
            8,
            badge_text.get_width() + badge_padding * 2,
            badge_text.get_height() + badge_padding,
        )
        badge_surface = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(badge_surface, (255, 255, 255, 200), badge_surface.get_rect(), border_radius=10)
        self.display.blit(badge_surface, badge_rect.topleft)
        self.display.blit(
            badge_text,
            (badge_rect.x + badge_padding, badge_rect.y + badge_padding // 2),
        )

        # Draw trail
        for index, block in enumerate(self.trail):
            alpha = max(20, 140 - index * 10)
            trail_surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(trail_surface, (120, 170, 160, alpha), trail_surface.get_rect(), border_radius=10)
            self.display.blit(
                trail_surface,
                (int(block.x * BLOCK_SIZE), int(block.y * BLOCK_SIZE)),
            )

        # Draw food with a soft glow
        food_center = (
            int(self.food.x * BLOCK_SIZE + BLOCK_SIZE / 2),
            int(self.food.y * BLOCK_SIZE + BLOCK_SIZE / 2),
        )
        glow_surface = pygame.Surface((BLOCK_SIZE * 3, BLOCK_SIZE * 3), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (255, 120, 170, 90), (BLOCK_SIZE * 1.5, BLOCK_SIZE * 1.5), int(BLOCK_SIZE * 0.75))
        pygame.draw.circle(glow_surface, (255, 140, 190, 140), (BLOCK_SIZE * 1.5, BLOCK_SIZE * 1.5), int(BLOCK_SIZE * 0.55))
        self.display.blit(
            glow_surface,
            (food_center[0] - int(BLOCK_SIZE * 1.5), food_center[1] - int(BLOCK_SIZE * 1.5)),
        )
        pygame.draw.circle(self.display, self.food_color, food_center, int(BLOCK_SIZE * 0.45))
        pygame.draw.circle(self.display, (255, 255, 255), food_center, int(BLOCK_SIZE * 0.48), 2)
        pygame.draw.circle(
            self.display,
            (255, 255, 255),
            (food_center[0] - 6, food_center[1] - 6),
            int(BLOCK_SIZE * 0.12),
        )

        # Draw snake with rounded blocks and soft shadow
        for i, block in enumerate(self.snake):
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            rect = pygame.Rect(
                int(block.x * BLOCK_SIZE),
                int(block.y * BLOCK_SIZE),
                BLOCK_SIZE,
                BLOCK_SIZE,
            )
            shadow_surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, (0, 0, 0, 50), shadow_surface.get_rect(), border_radius=10)
            self.display.blit(shadow_surface, (rect.x + 3, rect.y + 3))
            pygame.draw.rect(self.display, color, rect, border_radius=10)

        # UI text
        score_text = f"Score: {game_state.score}"
        mode_text = f"Mode: {game_state.mode}"
        epsilon_text = f"Epsilon: {game_state.epsilon:.3f}"
        best_text = f"Best: {game_state.best_score}"
        combo_text = f"Combo: x{game_state.combo}"
        speed_text = f"Speed: {game_state.manual_speed}"
        food_text = f"Food: {game_state.food_type}"
        action_name = ACTION_NAMES.get(game_state.last_action, "-")
        reward_text = f"Reward: {game_state.last_reward:+.2f}"
        action_text = f"Action: {action_name}"
        if game_state.last_state is None:
            state_text = "State: -"
        else:
            state_text = "State: [" + ",".join(str(int(x)) for x in game_state.last_state) + "]"
        lines = [
            score_text,
            best_text,
            mode_text,
            speed_text,
            epsilon_text,
            action_text,
            reward_text,
            combo_text,
            food_text,
            state_text,
        ]
        if game_state.mode == "training":
            lines.append(f"Episode: {game_state.episode}")
            lines.append(f"Avg Score: {game_state.avg_score:.2f}")
        if game_state.mode == "manual" and not game_state.manual_started:
            lines.append("Press arrow key to start")

        rendered = [self.font.render(line, True, TEXT_COLOR) for line in lines]
        max_width = max(text.get_width() for text in rendered)
        total_height = sum(text.get_height() for text in rendered) + (len(rendered) - 1) * 2
        padding = 10
        card_rect = pygame.Rect(6, 6, max_width + padding * 2, total_height + padding * 2)
        card_surface = pygame.Surface(card_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(card_surface, (255, 255, 255, 180), card_surface.get_rect(), border_radius=12)
        self.display.blit(card_surface, card_rect.topleft)

        y_offset = card_rect.y + padding
        for text in rendered:
            self.display.blit(text, (card_rect.x + padding, y_offset))
            y_offset += text.get_height() + 2

        if self.game_over:
            over_text = self.big_font.render("GAME OVER", True, OVERLAY_COLOR)
            score_text = self.font.render(f"Final Score: {self.score}", True, OVERLAY_COLOR)
            hint_text = self.font.render("Press Enter to play again", True, OVERLAY_COLOR)
            self.display.blit(
                over_text,
                (
                    WINDOW_SIZE // 2 - over_text.get_width() // 2,
                    WINDOW_SIZE // 2 - over_text.get_height(),
                ),
            )
            self.display.blit(
                score_text,
                (
                    WINDOW_SIZE // 2 - score_text.get_width() // 2,
                    WINDOW_SIZE // 2 + 10,
                ),
            )
            self.display.blit(
                hint_text,
                (
                    WINDOW_SIZE // 2 - hint_text.get_width() // 2,
                    WINDOW_SIZE // 2 + 32,
                ),
            )

        pygame.display.flip()

    def save_screenshot(self, file_name):
        pygame.image.save(self.display, file_name)

    def play_sound(self, name):
        if self.sounds and name in self.sounds:
            self.sounds[name].play()

    def _init_audio(self):
        self.sounds = {}
        if HEADLESS:
            self.sounds = None
            return
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1)
        except pygame.error:
            self.sounds = None
            return

        self.sounds = {
            "eat": self._make_tone(540, 0.08, 0.4),
            "bonus": self._make_tone(720, 0.1, 0.45),
            "poison": self._make_tone(240, 0.12, 0.5),
            "death": self._make_tone(120, 0.2, 0.6),
            "pause": self._make_tone(360, 0.08, 0.35),
        }

    def _make_tone(self, freq, duration, volume):
        sample_rate = 22050
        length = int(sample_rate * duration)
        buf = array("h")
        amplitude = int(32767 * volume)
        for i in range(length):
            value = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
            buf.append(value)
        return pygame.mixer.Sound(buffer=buf.tobytes())

    def _build_background(self):
        background = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        # Soft vertical gradient for a dreamy look
        for y in range(WINDOW_SIZE):
            t = y / (WINDOW_SIZE - 1)
            r = int(BG_TOP[0] * (1 - t) + BG_BOTTOM[0] * t)
            g = int(BG_TOP[1] * (1 - t) + BG_BOTTOM[1] * t)
            b = int(BG_TOP[2] * (1 - t) + BG_BOTTOM[2] * t)
            pygame.draw.line(background, (r, g, b), (0, y), (WINDOW_SIZE, y))

        # Subtle grid overlay
        for x in range(0, WINDOW_SIZE, BLOCK_SIZE):
            pygame.draw.line(background, GRID_COLOR, (x, 0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, BLOCK_SIZE):
            pygame.draw.line(background, GRID_COLOR, (0, y), (WINDOW_SIZE, y))

        # Light noise texture
        for _ in range(2500):
            x = random.randint(0, WINDOW_SIZE - 1)
            y = random.randint(0, WINDOW_SIZE - 1)
            background.set_at((x, y), (255, 230, 240))

        return background

    def _draw_background(self):
        self.display.blit(self.background, (0, 0))

        # Fade-in overlay on start for a gentle entrance
        elapsed = time.time() - self.start_time
        if elapsed < 1.0:
            alpha = int(200 * (1 - elapsed))
            overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            overlay.fill((255, 235, 245, alpha))
            self.display.blit(overlay, (0, 0))


def plot_training(scores):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not scores:
        return

    episodes = list(range(1, len(scores) + 1))
    window = 10
    averages = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        averages.append(sum(scores[start : i + 1]) / (i - start + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, scores, color="gray", alpha=0.5, label="Score")
    plt.plot(episodes, averages, color="green", linewidth=2, label="Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150)
    plt.show()


def handle_events(game_state):
    action = None
    toggle_pause = False
    switch_mode = False
    mode_select = None
    reset = False
    restart = False
    quit_game = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_game = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                quit_game = True
            if event.key == pygame.K_SPACE:
                toggle_pause = True
            if event.key == pygame.K_t or getattr(event, "unicode", "").lower() == "t":
                switch_mode = True
            if event.key == pygame.K_F1:
                mode_select = "manual"
            if event.key == pygame.K_F2:
                mode_select = "random_ai"
            if event.key == pygame.K_F3:
                mode_select = "training"
            if event.key == pygame.K_F4:
                mode_select = "test"
            if event.key == pygame.K_r:
                reset = True
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_1:
                game_state.manual_speed = 1
            if event.key == pygame.K_2:
                game_state.manual_speed = 2
            if event.key == pygame.K_3:
                game_state.manual_speed = 3
            if game_state.mode == "manual":
                if event.key == pygame.K_UP:
                    action = 1
                    game_state._manual_dir = directions[0]
                    game_state.manual_started = True
                elif event.key == pygame.K_RIGHT:
                    action = 1
                    game_state._manual_dir = directions[1]
                    game_state.manual_started = True
                elif event.key == pygame.K_DOWN:
                    action = 1
                    game_state._manual_dir = directions[2]
                    game_state.manual_started = True
                elif event.key == pygame.K_LEFT:
                    action = 1
                    game_state._manual_dir = directions[3]
                    game_state.manual_started = True

    return action, toggle_pause, switch_mode, mode_select, reset, restart, quit_game


def apply_mode_change(mode_name, game_state, agent):
    game_state.mode = mode_name
    if mode_name == "test":
        agent.load()
        agent.epsilon = 0.0
    if mode_name == "training":
        game_state.episode = 0
        game_state.avg_score = 0.0
        agent.epsilon = 0.8
        return True
    if mode_name == "manual":
        game_state.manual_started = False
    return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device)
    game = SnakeGameAI()

    game_state = GameState()
    game_state.manual_speed = 1
    game_state.best_score = load_best_score()
    modes = ["manual", "random_ai", "training", "test"]
    mode_index = 0

    paused = False
    training_scores = []
    training_done = False

    if HEADLESS:
        mode_index = modes.index("training")
        apply_mode_change("training", game_state, agent)
        training_done = False
        training_scores = []

    while True:
        action, toggle_pause, switch_mode, mode_select, reset, restart, quit_game = handle_events(game_state)

        if quit_game:
            break

        if toggle_pause:
            paused = not paused
            if paused:
                game.play_sound("pause")

        if mode_select:
            mode_index = modes.index(mode_select)
            reset_training = apply_mode_change(mode_select, game_state, agent)
            if reset_training:
                training_done = False
                training_scores = []
        elif switch_mode:
            mode_index = (mode_index + 1) % len(modes)
            reset_training = apply_mode_change(modes[mode_index], game_state, agent)
            if reset_training:
                training_done = False
                training_scores = []

        if reset:
            game.reset()
            if game_state.mode == "manual":
                game_state.manual_started = False

        if restart and game.game_over:
            game.game_over = False
            game.reset()

        if game.game_over:
            action, toggle_pause, switch_mode, mode_select, reset, restart, quit_game = handle_events(game_state)
            if quit_game:
                break
            if mode_select:
                mode_index = modes.index(mode_select)
                reset_training = apply_mode_change(mode_select, game_state, agent)
                if reset_training:
                    training_done = False
                    training_scores = []
            elif switch_mode:
                mode_index = (mode_index + 1) % len(modes)
                reset_training = apply_mode_change(modes[mode_index], game_state, agent)
                if reset_training:
                    training_done = False
                    training_scores = []
            if game_state.mode != "manual":
                if pygame.time.get_ticks() - game.game_over_time >= 2000:
                    game.game_over = False
                    game.reset()
            game.draw(game_state)
            tick_rate = FPS_LEVELS[game_state.manual_speed] if game_state.mode == "manual" else FPS_AI
            game.clock.tick(tick_rate)
            continue

        if paused:
            game.draw(game_state)
            tick_rate = FPS_LEVELS[game_state.manual_speed] if game_state.mode == "manual" else FPS_AI
            game.clock.tick(tick_rate)
            continue

        if game_state.mode == "manual" and not game_state.manual_started:
            game.draw(game_state)
            game.clock.tick(FPS_LEVELS[game_state.manual_speed])
            continue

        if game_state.mode == "training" and not training_done:
            abort_training = False
            for episode in range(300):
                action, toggle_pause, switch_mode, mode_select, reset, restart, quit_game = handle_events(game_state)
                if quit_game:
                    pygame.quit()
                    return
                if mode_select:
                    mode_index = modes.index(mode_select)
                    reset_training = apply_mode_change(mode_select, game_state, agent)
                    if reset_training:
                        training_done = False
                        training_scores = []
                    abort_training = True
                    break
                if switch_mode:
                    mode_index = (mode_index + 1) % len(modes)
                    reset_training = apply_mode_change(modes[mode_index], game_state, agent)
                    if reset_training:
                        training_done = False
                        training_scores = []
                    abort_training = True
                    break

                game.reset()
                done = False
                score = 0

                while not done:
                    action, toggle_pause, switch_mode, mode_select, reset, restart, quit_game = handle_events(game_state)
                    if quit_game:
                        pygame.quit()
                        return
                    if mode_select:
                        mode_index = modes.index(mode_select)
                        reset_training = apply_mode_change(mode_select, game_state, agent)
                        if reset_training:
                            training_done = False
                            training_scores = []
                        abort_training = True
                        done = True
                        continue
                    if switch_mode:
                        mode_index = (mode_index + 1) % len(modes)
                        reset_training = apply_mode_change(modes[mode_index], game_state, agent)
                        if reset_training:
                            training_done = False
                            training_scores = []
                        abort_training = True
                        done = True
                        continue

                    state = game.get_state()
                    game_state.last_state = state
                    action = agent.get_action(state)
                    next_state, reward, done, score = game.play_step(action)
                    agent.remember(state, action, reward, next_state, done)
                    agent.train_step()

                agent.decay_epsilon()
                training_scores.append(score)
                game_state.episode = episode + 1
                game_state.avg_score = sum(training_scores) / len(training_scores)
                game_state.epsilon = agent.epsilon
                game_state.last_action = action
                game_state.last_reward = reward
                game_state.last_state = game.get_state()
                game_state.combo = game.combo
                game_state.food_type = game.food_type

                if (episode + 1) % 50 == 0:
                    agent.save()
                    print(f"Episode {episode + 1} | Score {score} | Avg {game_state.avg_score:.2f}")
                    if not HEADLESS:
                        pygame.event.pump()
                        game.draw(game_state)
                        shot_name = f"training_ep_{episode + 1:03d}.png"
                        game.save_screenshot(shot_name)

            if abort_training:
                continue

            agent.save()
            training_done = True
            plot_training(training_scores)
            if HEADLESS:
                pygame.quit()
                return
            continue

        if game_state.mode == "manual":
            game_state.last_state = game.get_state()
            if hasattr(game_state, "_manual_dir"):
                # Prevent 180-degree turns
                if game_state._manual_dir + game.direction != pygame.Vector2(0, 0):
                    game.direction = game_state._manual_dir
            action = 1
        elif game_state.mode == "random_ai":
            game_state.last_state = game.get_state()
            action = random.randint(0, 2)
        elif game_state.mode == "test":
            state = game.get_state()
            game_state.last_state = state
            action = agent.get_action(state, greedy=True)
        else:
            state = game.get_state()
            game_state.last_state = state
            action = agent.get_action(state)

        next_state, reward, done, score = game.play_step(action)
        game_state.combo = game.combo
        game_state.food_type = game.food_type
        game_state.score = score
        game_state.epsilon = agent.epsilon
        game_state.last_action = action
        game_state.last_reward = reward
        if score > game_state.best_score:
            game_state.best_score = score
            save_best_score(score)

        if game_state.mode in ["training"]:
            agent.remember(game.get_state(), action, reward, next_state, done)
            agent.train_step()

        if done:
            game.game_over = True
            game.game_over_time = pygame.time.get_ticks()
            # Throttle screenshots to avoid spamming files.
            if game_state.mode != "manual":
                now = time.time()
                if now - game.last_game_over_shot_time >= 30:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    game.save_screenshot(f"game_over_{timestamp}.png")
                    game.last_game_over_shot_time = now
            if not game.game_over_sound_played:
                game.play_sound("death")
                game.game_over_sound_played = True
        else:
            if game_state.mode in ["manual", "test"]:
                if game.last_event == "eat":
                    game.play_sound("eat")
                elif game.last_event == "bonus":
                    game.play_sound("bonus")
                elif game.last_event == "poison":
                    game.play_sound("poison")

        game.draw(game_state)
        tick_rate = FPS_LEVELS[game_state.manual_speed] if game_state.mode == "manual" else FPS_AI
        game.clock.tick(tick_rate)

    pygame.quit()


if __name__ == "__main__":
    main()
