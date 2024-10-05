# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import cv2
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

import mmcv
from mmcv.transforms import Compose
from mmengine.runner.checkpoint import get_state_dict, save_checkpoint
from mmengine.utils import track_iter_progress


class TimerData:
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.reset()

    def reset(self):
        self.counter = 0
        self.accumulated_time = 0


class Timer:
    def __init__(
        self,
        timer_data: TimerData,
        batch_size: int = 1,
        print_interval: int = 25,
        start_with_carriage_return: bool = True,
    ):
        self.print_interval = print_interval
        self.batch_size = batch_size
        self.start_with_carriage_return = start_with_carriage_return
        self.timer_data = timer_data

    def reset(self):
        self.timer_data.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, maybe_print: bool = True):
        end_time = time.time()
        self.timer_data.counter += 1
        self.timer_data.accumulated_time += end_time - self.start_time
        if maybe_print:
            self.maybe_print_time()

    def maybe_print_time(self, reset: bool = True):
        if self.timer_data.counter % self.print_interval == 0:
            frame_count = self.timer_data.counter * self.batch_size
            fps = frame_count / self.timer_data.accumulated_time
            cr = "\n"
            print(
                f"{cr if self.start_with_carriage_return else ''}Processed {self.timer_data.counter} frames at {fps:.2f} fps"
            )
            if reset:
                self.reset()

    def __enter__(self):
        self.start()

    def __exit__(self, *args, **kwargs):
        self.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="MMDetection video demo")
    parser.add_argument("video", help="Video file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="Bbox score threshold"
    )
    parser.add_argument("--out", type=str, help="Output video file")
    parser.add_argument("--show", action="store_true", help="Show video")
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Save a copy of the checkpoint",
    )
    parser.add_argument(
        "--wait-time",
        type=float,
        default=1,
        help="The interval of show (s), 0 is block",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # assert args.out or args.show, \
    #     ('Please specify at least one operation (save/show the '
    #      'video) with the argument "--out" or "--show"')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if args.save_checkpoint:
        state_dict = get_state_dict(model)
        save_checkpoint(state_dict, args.save_checkpoint)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = "mmdet.LoadImageFromNDArray"
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.out,
            fourcc,
            video_reader.fps,
            (video_reader.width, video_reader.height),
        )

    timer_data = TimerData()

    for frame in track_iter_progress((video_reader, len(video_reader))):
        with Timer(timer_data=timer_data, batch_size=1):
            result = inference_detector(model, frame, test_pipeline=test_pipeline)
            # labels = result.pred_instances.labels.cpu()
            # bboxes = result.pred_instances.bboxes.cpu()
        if torch.is_floating_point(result.pred_instances.labels):
            result.pred_instances.labels = result.pred_instances.labels.to(torch.int64)
        if args.out or args.show:
            visualizer.add_datasample(
                name="video",
                image=frame,
                data_sample=result,
                draw_gt=False,
                show=False,
                pred_score_thr=args.score_thr,
            )
            frame = visualizer.get_image()

            if args.show:
                cv2.namedWindow("video", 0)
                mmcv.imshow(frame, "video", args.wait_time)
            if args.out:
                video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
