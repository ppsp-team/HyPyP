import tracemalloc
import time

class MemoryMonitor(object):
    def __init__(self):
        """
        Track memory usage during a code block execution. Usage:

        ```
        with MemoryMonitor():
            // my code
        ```

        """
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
    def __init__(self, label:str='time_tracker'):
        """
        Track code execution time. Usage:

        ```
        with TimeTracker('foo'):
            // my code
        ```

        Args:
            label (str): identifier for the code block (for display only)
        """
        self.start_time = None
        self.stop_time = None
        self.duration = None
        self.label = label

    @staticmethod
    def human_readable_duration(seconds):
        if seconds < 5:
            return f"{seconds:.2f} seconds"
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        if seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        days = seconds / 86400
        return f"{days:.1f} days"

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.duration = self.stop_time - self.start_time

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()
        print(f"--- [{self.label}] {self.duration} seconds ---")