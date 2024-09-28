import numpy as np
import cv2
import decord
from decord import VideoReader

BASE_FPS = 24
BASE_RGB_MAX = 255
ABERAGE_SECONDS_PER_ACTION = 5

def video_mixer(
        vr : VideoReader, start_time=0, end_time=None,
        threshold=BASE_RGB_MAX/2, frame_limit=BASE_FPS*ABERAGE_SECONDS_PER_ACTION, step_frames=3
        ):
    """
    Mixes frames from a video by averaging them until the difference between two
    consecutive frames is larger than the given threshold.

    Args:
        vr (decord.VideoReader): The video to mix.
        start_time (int, optional): The start time for mixing. Defaults to 0.
        end_time (int, optional): The end time for mixing. Defaults to None.
        threshold (int, optional): The threshold for mixing. Defaults to 120.
        frame_limit (int, optional): The maximum number of frames to mix. Defaults to 100.
        step_frames (int, optional): The step size for mixing. Defaults to 5.

    Yields:
        numpy.ndarray: A mixed frame.
    """
    if end_time is None:
        end_time = len(vr)

    frame_id_list = range(start_time, end_time, step_frames)
    mixed_frame, frame_iter = None, 0
    batch = vr.get_batch(frame_id_list)
    for frame in batch:
        frame = frame.numpy()
        if mixed_frame is None:
            mixed_frame, frame_iter = frame, 1
            continue
        if frame_iter >= frame_limit:
            yield mixed_frame
            mixed_frame, frame_iter = frame, 1
            continue

        frame_iter += 1
        # diff = abs(frame.numpy() - mixed_frame.numpy())
        diff = abs(frame - mixed_frame)

        if diff.mean() > threshold:
            # yield torch.tensor(mixed_frame)
            yield mixed_frame
            mixed_frame, frame_iter = frame, 1
            continue

        # mixed_frame = cv2.addWeighted(mixed_frame.numpy(), 0.5, frame.numpy(), 0.5, 0)
        mixed_frame = cv2.addWeighted(mixed_frame, 0.5, frame, 0.5, 0)
        # mixed_frame = torch.tensor(mixed_frame)
    yield mixed_frame


