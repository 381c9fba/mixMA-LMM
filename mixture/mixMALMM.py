import numpy as np
import cv2
import decord
import torch
from decord import VideoReader

from lavis.models import load_model_and_preprocess

BASE_FPS = 24
BASE_RGB_MAX = 255
AVERAGE_SECONDS_PER_ACTION = 5


class MALMMInferencePipeline:
    def __init__(self, preprocess_fn = None, postprocess_fn = None,
                 num_frames = 20):
        """
        Initialize the inference pipeline.

        Args:
            model (object): The trained model to use for inference.
            preprocess_fn (callable): A function to preprocess the input data.
            postprocess_fn (callable): A function to postprocess the output data.
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct_malmm", model_type="vicuna7b",
            is_eval=True, device=self.device, memory_bank_length=100, num_frames=num_frames,
        )

        if preprocess_fn is None:
            self.preprocess_fn = self.preprocess_fn
        else:
            self.preprocess_fn = preprocess_fn

        if postprocess_fn is None:
            self.postprocess_fn = (lambda x: x)
        else:
            self.postprocess_fn = postprocess_fn

    def run(self, input_data,
            prompt="Describe what's happening in this video. Task: Captioning video. Answer:"):
        """
        Run the inference pipeline on the input data.

        Args:
            vr (decord.VideoReader): The input data to process.

        Returns:
            list: The output data after running the pipeline.
        """
        self.caption_frequency = 20 # before MOCK, value only for assign

        # Preprocess the input data
        preprocessed_data = self.preprocess_fn(input_data)

        # Run the model on the preprocessed data
        output_data = self.model.generate(
            {"image": preprocessed_data, "prompt": prompt}, num_captions=self.caption_frequency)

        # Postprocess the output data
        # postprocessed_data = self.postprocess_fn(output_data)

        # ['cooking', 'recipe', 'egg recipe', 'cooking eggs', 'cooking eggs on stove']        #
        return output_data

    def preprocess_fn(self, vr):
        """
        Preprocess the input data.

        Args:
            vr (decord.VideoReader): The input data to process.

        Returns:
            object: The preprocessed input data.
        """
        gen = self.video_mixer(vr)
        mix_lst, mean_id_mixes = zip(*gen)

        self.caption_frequency = len(mix_lst) # MOCK: 1 caption per 1 mixframe
        self.mean_id_mixes = mean_id_mixes # MOCK

        tensors_mixs = self.to_tensor(mix_lst)
        video = self.vis_processors["eval"](tensors_mixs).to(self.device).unsqueeze(0)

        return video

    def video_mixer(
            self,
            vr : VideoReader, start_time=0, end_time=None,
            threshold=BASE_RGB_MAX/2, frame_limit=BASE_FPS*AVERAGE_SECONDS_PER_ACTION, step_frames=3
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

    def linspace_extrat_frames(vr, start_time, end_time, fps, num_frames=100):
        """
        Extracts frames from a video given the start and end time, and the frame rate.

        Args:
            vr (decord.VideoReader): The video to extract frames from.
            start_time (int): The start time for extracting frames.
            end_time (int): The end time for extracting frames.
            fps (int): The frame rate of the video.
            num_frames (int, optional): The number of frames to extract. Defaults to 100.

        Returns:
            torch.Tensor: A tensor of the extracted frames with shape (num_frames, H, W, 3).
        """
        # ПРИМЕЧАНИЕ: это дефолтный препроцессор видео, использовать если video_mixer нерабочий

        start_index = int(round(start_time * fps))
        end_index = int(round(end_time * fps))
        select_frame_index = np.rint(np.linspace(start_index, end_index-1, num_frames)).astype(int).tolist()
        frames = vr.get_batch(select_frame_index).permute(3, 0, 1, 2).to(torch.float32)
        return frames

    def to_tensor(mix_list :tuple) -> torch.Tensor:
        """
        Converts a tuple of mixed frames to a tensor.

        Args:
            mix_list (tuple): A tuple of mixed frames with shape (H, W, 3).

        Returns:
            torch.Tensor: A tensor of the mixed frames with shape (3, H, W).
        """
        return torch.tensor(np.array(mix_list)).permute(3, 0, 1, 2).to(torch.float32)


