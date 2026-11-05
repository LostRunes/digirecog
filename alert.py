"""
Attention beeper.  Beeps until you press  Ctrl + Space  (works globally, even
if another window is focused).

Usage (from code / terminal):
    python alert.py "Handwritten Digit Recognition" "Need your decision on X"
"""
import sys, time, ctypes

try:
    import winsound
except ImportError:  # non-Windows fallback
    winsound = None

VK_CONTROL, VK_SPACE = 0x11, 0x20


def pressed(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def beep():
    if winsound:
        winsound.Beep(1200, 250); winsound.Beep(900, 250)
    else:
        print("\a", end="", flush=True); time.sleep(0.5)


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else "Unknown project"
    msg = sys.argv[2] if len(sys.argv) > 2 else "Claude needs your input."
    print("=" * 60)
    print(f"  ATTENTION NEEDED  |  Project: {project}")
    print(f"  {msg}")
    print("  Press  CTRL + SPACE  to stop this alert.")
    print("=" * 60, flush=True)
    while not (pressed(VK_CONTROL) and pressed(VK_SPACE)):
        beep()
        for _ in range(10):                      # ~0.5s pause, still responsive
            if pressed(VK_CONTROL) and pressed(VK_SPACE):
                break
            time.sleep(0.05)
    print("Alert stopped.")


if __name__ == "__main__":
    main()
