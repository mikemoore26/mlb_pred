# watch_and_push.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess, time, os, sys

TASK = "On-save review: spot bugs, fragile code, and missing tests."
WATCH_PATH = "."

class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        # ignore temp/save artifacts
        name = os.path.basename(event.src_path)
        if name.startswith(".") or name.endswith("~"):
            return
        time.sleep(0.2)  # debounce
        print(f"Detected change: {event.src_path}")
        subprocess.run(
            [sys.executable, "push_project.py", "--only_changed", "--task", TASK],
            check=False
        )

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception:
        pass
    obs = Observer()
    obs.schedule(Handler(), WATCH_PATH, recursive=True)
    obs.start()
    print("Watching for changes… Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
