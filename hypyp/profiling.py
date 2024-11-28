import tracemalloc
import time

class MemoryMonitor(object):
    def __init__(self):
        pass
    
    def format_memory_size(self, size_bytes):
        if size_bytes >= 1 << 30:  # Greater than or equal to 1 GiB
            return f"{size_bytes / (1 << 30):.2f} G"
        elif size_bytes >= 1 << 20:  # Greater than or equal to 1 MiB
            return f"{size_bytes / (1 << 20):.2f} M"
        elif size_bytes >= 1 << 10:  # Greater than or equal to 1 KiB
            return f"{size_bytes / (1 << 10):.2f} k"
        else:  # Less than 1 KiB
            return f"{size_bytes} B"

    def __enter__(self):
        tracemalloc.start()

    def __exit__(self, *args):
        res = tracemalloc.get_traced_memory()
        print(f"[MemoryMonitor] allocated: {self.format_memory_size(res[0])}, peak: {self.format_memory_size(res[1])}", )
        tracemalloc.stop()

class TimeTracker(object):
    def __init__(self, label):
        self.start_time = None
        self.stop_time = None
        self.label = label

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        self.stop_time = time.time()
        duration = self.stop_time - self.start_time
        print(f"--- [{self.label}] {duration} seconds ---")