import torch
from collections import defaultdict
from tqdm import tqdm

class CudaProfiler():
    """`
    A utility class to measure execution times of multiple, named segments
    using torch.cuda.Event.
    """
    _instance = None

    # Singleton design
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, use_prof=False, device=None):
        self.use_prof = use_prof  # Set "False" if users want to turn this profiler off.
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialized = True

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Cannot use profiler.")
            self.device = device if device is not None else torch.device("cuda")

            # Store events for each named segment
            # Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]]
            self._events = defaultdict(lambda: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)
            ))
            self._started_segments = set()  # To track if a segment has been started
            self.num_iter = 0
            self.l_prof = {}
            self.l_cntStart = {}

    def start(self, segment_name: str):
        """Records the start event for a given segment."""
        if segment_name in self._started_segments:
            raise ValueError(f"Segment '{segment_name}' has already been started without being stopped.")

        # Access or create events for this segment
        start_event, _ = self._events[segment_name]
        start_event.record()
        self._started_segments.add(segment_name)
        self.l_prof.setdefault(segment_name, []).append(0)
        if segment_name in self.l_cntStart:
            self.l_cntStart[segment_name] += 1
        else:
            self.l_cntStart[segment_name] = 1

        torch.cuda.nvtx.range_push(segment_name)

    def stop(self, segment_name: str):
        """Records the end event for a given segment."""
        if segment_name not in self._started_segments:
            raise ValueError(f"Segment '{segment_name}' was stopped without being started.")

        _, end_event = self._events[segment_name]
        end_event.record()
        self._started_segments.remove(segment_name)

        torch.cuda.nvtx.range_pop()

    def format_string_fixed_width(self, text, width, alignment='<'):
        """
        Formats a string to a fixed width, padding with spaces if shorter
        and truncating if longer.

        Args:
            text: The value to format. It will be converted to a string.
            width (int): The desired total width (number of characters) for the output string.
            alignment (str, optional): The alignment direction. Defaults to '<'.
                                    '<': Left-align (pads spaces on the right, truncates from the right).
                                    '>': Right-align (pads spaces on the left, truncates from the left).
                                    '^': Center-align (pads spaces on both sides, truncates from both sides).

        Returns:
            str: The formatted string, adjusted to the specified width.
        """
        # f-string formatting specification:
        # {text}       : The value to be formatted
        # :            : Indicates the start of the format specifier
        # {alignment}  : Alignment direction ('<', '>', '^')
        # {width}      : The total width of the field
        # .{width}     : Maximum length, truncates the string if it exceeds this width.

        # f-strings automatically convert the value to a string.
        return f"{text:{alignment}{width}.{width}}"

    def synchronize_all(self, rank):
        """
        Synchronize all the PyTorch CUDA kernel calls.
        Print the measured execution time all at once.
        """
        if rank == 0:
            # Waits for all recorded events to complete.
            # Ensure all events are recorded before querying times
            torch.cuda.synchronize()

            self.num_iter += 1
            l_tags = list(self.l_prof)
            header = ""
            for tag in l_tags:
                header += self.format_string_fixed_width(tag, 10) + "|"
            tqdm.write(header)
            tqdm.write("-" * len(header))

            msg = ""
            for tag in l_tags:
                self.l_prof[tag].append(self.get_time_ms(tag))
                try:
                    avg_prof = (sum(self.l_prof[tag]) / self.num_iter) * self.l_cntStart[tag]
                    msg += self.format_string_fixed_width(f"{avg_prof:.4f}", 10) + "|"
                    self.l_cntStart[tag] = 0
                except Exception:
                    self.handleError("Error while printing out profiling results")
            tqdm.write(msg)

            # if self.num_iter > 30: # This can be used when viewing result with the NVIDIA Nsight system.
            #     exit()

    def get_time_ms(self, segment_name: str) -> float:
        """
        Returns the elapsed time for a given segment in milliseconds.
        Requires synchronize_all() or end_event.synchronize() for the specific segment
        to be called first.
        """
        if segment_name in self._started_segments:
            raise RuntimeError(f"Segment '{segment_name}' is still running. Call stop() and synchronize_all() first.")

        start_event, end_event = self._events[segment_name]
        if not start_event.query() or not end_event.query():  # Check if events completed
            raise RuntimeError(f"Events for segment '{segment_name}' have not completed. Call synchronize_all() first.")

        return start_event.elapsed_time(end_event)

    def reset(self):
        """Clears all recorded events and internal state."""
        self._events.clear()
        self._started_segments.clear()

    def get_all_times_ms(self) -> dict[str, float]:
        """Returns a dictionary of all recorded segment times."""
        self.synchronize_all()  # Ensure all times are ready
        times = {}
        for name in self._events:
            times[name] = self.get_time_ms(name)
        return times