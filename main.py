import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import random
import collections
import time
import os
from model import DQNBrain
from sans_env import SansEnv
from pynput import mouse, keyboard

MEMORY_SIZE = 150000
BATCH_SIZE = 512
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = 100000

coords = []
def on_click(x, y, button, pressed):
    if pressed:
        coords.append((x, y))
        print(f" Captured Point {len(coords)}: ({x}, {y})")
        if len(coords) >= 4:
            return False

print("\n[CALIBRATION] Follow exactly:")
print("1. Click TOP-LEFT of Battle Arena")
print("2. Click BOTTOM-RIGHT of Battle Arena")
print("3. Click TOP-LEFT of Yellow HP Bar")
print("4. Click BOTTOM-RIGHT of Yellow HP Bar")

while len(coords) < 4:
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    if len(coords) < 4:
        print("Point missed! Restarting calibration loop...")
        coords = []

mon = {"top": int(coords[0][1]), "left": int(coords[0][0]),
       "width": int(abs(coords[1][0]-coords[0][0])),
       "height": int(abs(coords[1][1]-coords[0][1]))}

hp_mon = {"top": int(coords[2][1]), "left": int(coords[2][0]),
          "width": int(abs(coords[3][0]-coords[2][0])),
          "height": int(abs(coords[3][1]-coords[2][1]))}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

print(f"\n[SYSTEM] DEVICE: {device}")
env = SansEnv(mon, hp_mon)
brain = DQNBrain(n_actions=5).to(device)

def safe_save(model, path="sans_brain.pth"):
    temp_path = path + ".tmp"
    try:
        torch.save(model.state_dict(), temp_path)
        os.replace(temp_path, path)
    except Exception as e:
        print(f"\n[SAVE ERROR] {e}")

if os.path.exists("sans_brain.pth"):
    try: brain.load("sans_brain.pth")
    except: pass

target_brain = DQNBrain(n_actions=5).to(device)
target_brain.load_state_dict(brain.state_dict())
optimizer = optim.Adam(brain.parameters(), lr=0.0002)
memory = collections.deque(maxlen=MEMORY_SIZE)

learning_active, manual_mode, death_count = True, False, 0

def on_press(key):
    global learning_active, manual_mode
    try:
        if key.char == 'p': learning_active = not learning_active
        if key.char == 'l': manual_mode = not manual_mode
    except AttributeError: pass

keyboard.Listener(on_press=on_press).start()

try:
    step_count = 0
    while True:
        if not learning_active or manual_mode:
            time.sleep(0.1)
            continue

        loop_start = time.time()
        state, _ = env.get_state()
        state = state.to(device, non_blocking=True)

        epsilon = max(EPSILON_END, EPSILON_START - (step_count / EPSILON_DECAY_STEPS))

        if random.random() < epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                with autocast():
                    action = brain(state).max(1)[1].item()

        next_state, reward, done, hp = env.step(action)

        if done:
            death_count += 1
        else:
            memory.append((state.cpu(), action, reward, next_state, done))

        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            s, a, r, ns, d = zip(*batch)
            s = torch.cat(s).to(device, non_blocking=True)
            ns = torch.cat(ns).to(device, non_blocking=True)
            a = torch.LongTensor(a).unsqueeze(1).to(device, non_blocking=True)
            r = torch.FloatTensor(r).to(device, non_blocking=True)
            d = torch.BoolTensor(d).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                curr_q = brain(s).gather(1, a)
                with torch.no_grad():
                    next_q = target_brain(ns).max(1)[0]
                    expected_q = r + (GAMMA * next_q * ~d)
                loss = F.smooth_l1_loss(curr_q.squeeze(), expected_q)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if step_count > 0 and step_count % 3000 == 0:
            target_brain.load_state_dict(brain.state_dict())
            safe_save(brain)

        step_count += 1
        print(f"BATTLE | Steps: {step_count} | Deaths: {death_count} | Eps: {epsilon:.2f} | HP: {hp*100:.0f}% ", end="\r")

except KeyboardInterrupt:
    env.release_all()
    safe_save(brain)
