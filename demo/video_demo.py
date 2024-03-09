# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch

import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--perf', action="store_true", help='Performance run')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show or args.perf, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_reader = mmcv.VideoReader(args.video)
    if not args.perf:
        video_writer = None
        if args.out:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                args.out, fourcc, video_reader.fps,
                (video_reader.width, video_reader.height))

    if args.perf:
        perf_run = True
        perf_counter = 0
        start_time = None

    for frame in mmcv.track_iter_progress(video_reader):
        
        frame = torch.from_numpy(frame).to(args.device)
        
        if args.perf:
            # Timing
            if perf_counter == 0:
                start_time = time.time()
            perf_counter += 1

        result = inference_detector(model, frame)
        
        if not args.perf:
            frame = model.show_result(frame, result, score_thr=args.score_thr)
            if args.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame, 'video', args.wait_time)
            if args.out:
                video_writer.write(frame)
        elif perf_counter == 20:
            stop_time = time.time()
            fps = perf_counter / (stop_time - start_time)
            print(f"\nfps={fps}")
            perf_counter = 0

    if not args.perf:
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
