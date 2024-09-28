import decord
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge('torch')

from mixture.mixMALMM import MALMMInferencePipeline

# main — usage example

example_video = "example/video.mp4"

if __name__ == "__main__":
    vr = VideoReader(example_video, ctx=cpu(0))
    print(MALMMInferencePipeline().run(vr))
