"""
forge/integration/watchdog.py

Resiliency daemon for Forge.
Restarts training jobs if they fail or are preempted.
"""

from __future__ import annotations
import subprocess
import time
import sys
from typing import List

class ForgeWatchdog:
    def __init__(self, command: List[str], max_retries: int = 10, backoff_base: int = 5):
        self.command = command
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.retries = 0

    def start(self):
        print(f"[Forge.Watch] Starting watchdog for command: {' '.join(self.command)}")
        
        while self.retries < self.max_retries:
            print(f"[Forge.Watch] Attempt {self.retries + 1}/{self.max_retries}")
            
            process = subprocess.Popen(self.command)
            
            try:
                exit_code = process.wait()
            except KeyboardInterrupt:
                print("[Forge.Watch] Interrupted by user. Shutting down...")
                process.terminate()
                sys.exit(0)
            
            if exit_code == 0:
                print("[Forge.Watch] Process finished successfully.")
                break
            else:
                print(f"[Forge.Watch] Process failed with exit code {exit_code}.")
                self.retries += 1
                if self.retries < self.max_retries:
                    wait_time = self.backoff_base * self.retries
                    print(f"[Forge.Watch] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("[Forge.Watch] Max retries reached. Giving up.")
                    sys.exit(exit_code)

def run_watchdog(command: List[str], max_retries: int = 10):
    watchdog = ForgeWatchdog(command, max_retries=max_retries)
    watchdog.start()
