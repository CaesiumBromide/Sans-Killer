import cv2
import numpy as np
import mss
import torch
import time
from pynput.keyboard import Controller, Key

class SansEnv:
    def __init__(self, monitor, hp_monitor):
        self.sct = mss.mss()
        self.monitor = monitor
        self.hp_monitor = hp_monitor
        self.keyboard = Controller()
        self.keys = [Key.left, Key.right, Key.up]
        self.battle_step = 0

    def get_hp_percentage(self):
        try:
            sct_img = self.sct.grab(self.hp_monitor)
            img = np.array(sct_img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

            active_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            hp_ratio = active_pixels / total_pixels

            return max(0.01, hp_ratio), active_pixels > (total_pixels * 0.05)
        except:
            return 0.5, False

    def get_state(self):
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
        blue_mask = cv2.inRange(hsv, (80, 100, 100), (110, 255, 255))
        danger = cv2.bitwise_or(white_mask, blue_mask)

        state = cv2.resize(danger, (64, 64), interpolation=cv2.INTER_NEAREST)

        _, is_alive = self.get_hp_percentage()
        is_dead = not is_alive

        return torch.FloatTensor(state).unsqueeze(0).unsqueeze(0) / 255.0, is_dead

    def recover_to_battle(self):
        print("\n[RECOVERY] HP Area Blackened. Restarting...")
        self.release_all()

        for _ in range(15):
            for key in ['z', Key.enter]:
                self.keyboard.press(key)
                time.sleep(0.1)
                self.keyboard.release(key)
                time.sleep(0.05)

        print("[RECOVERY] Returning to Sans...")
        start_time = time.time()
        while True:
            self.keyboard.press(Key.right)

            for key in ['z', 'x', Key.enter]:
                self.keyboard.press(key)
                time.sleep(0.02)
                self.keyboard.release(key)

            _, is_alive = self.get_hp_percentage()
            if is_alive:
                print("[RECOVERY] HP Detected. Fight Start.")
                self.keyboard.release(Key.right)
                break

            if time.time() - start_time > 60:
                self.keyboard.release(Key.right)
                break

    def step(self, action_idx):
        for k in self.keys: self.keyboard.release(k)

        if action_idx == 0: self.keyboard.press(Key.left)
        elif action_idx == 1: self.keyboard.press(Key.right)
        elif action_idx == 2: self.keyboard.press(Key.up)
        elif action_idx == 3:
            self.keyboard.press(Key.up)
            self.keyboard.press(Key.right)

        if self.battle_step % 8 == 0:
            self.keyboard.press('z')
            time.sleep(0.01)
            self.keyboard.release('z')

        self.battle_step += 1
        next_state, is_dead = self.get_state()
        current_hp, _ = self.get_hp_percentage()

        reward = 1.0 * current_hp
        if is_dead:
            reward = -100.0
            self.recover_to_battle()

        return next_state, reward, is_dead, current_hp

    def release_all(self):
        for k in [Key.left, Key.right, Key.up, 'z', 'x', Key.enter]:
            try: self.keyboard.release(k)
            except: pass
