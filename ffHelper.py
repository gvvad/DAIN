import subprocess
import json
import threading, queue
# import asyncio
# from sys import stdout
import numpy as np
import torch

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
FFPROBE_CMDL = [FFPROBE, "-v", "quiet", "-show_streams", "-select_streams" ,"v", "-of", "json=compact=1"]


class FFSource:
    deph = 3

    def __init__(self, src, fmt="rgb24", param=[]) -> None:
        self.proc = None
        self.src = src
        self.fmt = fmt
        self.param = param

        probeProc = subprocess.run(FFPROBE_CMDL + [src], stdout=subprocess.PIPE)
        ln = probeProc.stdout.decode("utf-8")
        self.streamInfo = json.loads(ln)["streams"][0]

        self.frameSize = self.streamInfo["height"] * self.streamInfo["width"] * self.deph

    def __iter__(self):
        if not self.proc:
            self.proc = subprocess.Popen(
                [FFMPEG,
                "-v", "error",
                "-i", self.src,
                ] + self.param + [
                "-an", "-sn",
                "-pix_fmt", self.fmt,
                "-f", "rawvideo",
                "pipe:1"],
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                # bufsize= self.frameSize * 8
            )
        return self

    def __next__(self):
        buf = self.proc.stdout.read(self.frameSize)

        if len(buf) != self.frameSize:
            self.proc.terminate()
            self.proc = None
            raise StopIteration
        
        arr = np.frombuffer(buf, dtype=np.uint8)
        arr = arr.reshape([self.streamInfo["height"], self.streamInfo["width"], self.deph])
        return arr


class FFDestination:
    deph = 3
    
    def __init__(self, dst, width, height, frameRate, fmt="rgb24", param=["-y", "-c:v", "v308"], queue_size=10) -> None:
        self.dst = dst
        self.width = width
        self.height = height
        self.fmt = fmt
        self.frameRate = frameRate

        self.frameSize = width * height * self.deph

        self.proc = subprocess.Popen(
            [FFMPEG,
            "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", self.fmt,
            "-s:v", f"{self.width}x{self.height}",
            "-r", self.frameRate,
            "-i", "pipe:0"] + param + [self.dst],
            stdin=subprocess.PIPE,
        )

        self.queue = queue.Queue(queue_size)
        self.thread = threading.Thread(target=self._writerWorker, daemon=False).start()   

    def _writerWorker(self):
        while True:
            res = self.queue.get()
            if isinstance(res, torch.Tensor):
                self.proc.stdin.write(np.array(res.cpu(), dtype=np.uint8).tobytes())
            elif isinstance(res, np.ndarray):
                self.proc.stdin.write(res.tobytes())
            else:
                self.proc.stdin.close()
                self.proc.communicate()
                self.proc = None
                break
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def putData(self, data:bytes):
        self.queue.put(data)

    def close(self):
        self.queue.put(None)
        self.queue.join

def frameRateMul(framerate, mul) -> str:
    try:
        return f"{float(framerate)*mul:.4f}"
    except ValueError:
        res = framerate.split("/")
        if len(res) != 2:
            raise ValueError(f"Cannot recognize frame rate: {framerate}")
        num = int(res[0])
        denom = int(res[1])

        return f"{num*int(mul)}/{denom}"