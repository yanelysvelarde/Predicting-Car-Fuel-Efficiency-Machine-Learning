import time

class Timer:
    def __init__(self, task_name="Task"):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if duration >= 60:
            minutes = int(duration // 60)
            seconds = round(duration % 60)
            print(f"⏱️  {self.task_name} completed in {minutes}m {seconds}s.\n")
        else:
            print(f"⏱️  {self.task_name} completed in {round(duration, 2)} seconds.\n")