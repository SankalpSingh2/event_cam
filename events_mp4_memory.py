import argparse
import os
import numpy as np
import glob
import cv2
import tqdm
import ffmpeg  # Make sure to use ffmpeg-python

def write_frames_to_video(npz_dir, output_video, frame_shape, framerate=60, vcodec='libx264'):
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    height, width = frame_shape

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .output(output_video, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    for npz_file in tqdm.tqdm(npz_files, desc="Processing frames"):
        data = np.load(npz_file)
        event_frame = np.zeros((height, width), dtype=np.uint8)

        if 'x' in data and 'y' in data and 'p' in data:
            x, y, p = data['x'], data['y'], data['p']
            event_frame[y, x] = np.where(p == 1, 255, 0)

        rgb_frame = cv2.cvtColor(event_frame, cv2.COLOR_GRAY2RGB)

        try:
            process.stdin.write(rgb_frame.astype(np.uint8).tobytes())
        except BrokenPipeError:
            print("Error: Broken pipe. FFmpeg may have encountered an issue.")
            break

    process.stdin.close()
    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directory of .npz files to mp4")
    parser.add_argument("--npz_dir", "-i", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--output_video", "-o", type=str, required=True, help="Output mp4 file path")
    parser.add_argument("--framerate", "-f", type=int, default=60, help="Frame rate of the output video")
    parser.add_argument("--frame_shape", "-s", type=int, nargs=2, required=True, help="Shape of the frames as (height, width)")
    args = parser.parse_args()

    write_frames_to_video(args.npz_dir, args.output_video, tuple(args.frame_shape), framerate=args.framerate)
