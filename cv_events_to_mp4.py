import argparse
import os
import numpy as np
import glob
import cv2
import tqdm
from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

def write_frames_to_video(npz_dir, output_video, framerate=15, size=(96, 96), max_frames=None):
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in directory: {npz_dir}")
        return

    print(f"Detected {len(npz_files)} .npz files in directory: {npz_dir}")

    # Open the VideoWriter with specified parameters
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)

    frame_count = 0

    for npz_file in tqdm.tqdm(npz_files, desc="Processing frames"):
        data = np.load(npz_file)

        if 'x' in data and 'y' in data and 'p' in data:
            # Initialize a new frame for each file
            event_frame = np.zeros(size, dtype=np.uint8)

            x, y, p = data['x'], data['y'], data['p']
            for i in range(len(x)):
                if p[i] == 1:
                    if 0 <= y[i] < size[1] and 0 <= x[i] < size[0]:
                        event_frame[y[i], x[i]] = 255

            # Convert to RGB and resize the frame
            rgb_frame = cv2.cvtColor(event_frame, cv2.COLOR_GRAY2RGB)
            rgb_frame_resized = cv2.resize(rgb_frame, size)

            # Write the frame to the video file
            out.write(rgb_frame_resized)
            frame_count += 1

            if max_frames is not None and frame_count >= max_frames:
                break

    out.release()  # Release the VideoWriter

    # Now, apply speedup effect using moviepy
    clip = VideoFileClip(output_video)
    final = clip.fx(vfx.speedx, 3)
    print("fps after speedup: {}".format(final.fps))
    final.write_videofile(output_video, codec="libx264")

    print(f"Video saved to {output_video} with {frame_count} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directory of .npz files to mp4")
    parser.add_argument("--npz_dir", "-i", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--output_video", "-o", type=str, required=True, help="Output mp4 file path")
    parser.add_argument("--framerate", "-f", type=int, default=15, help="Frame rate of the output video")
    parser.add_argument("--size", "-s", type=int, nargs=2, default=(96, 96), help="Width and Height of the output video")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Maximum number of frames to write to the video")
    args = parser.parse_args()

    write_frames_to_video(args.npz_dir, args.output_video, framerate=args.framerate, size=args.size, max_frames=args.max_frames)
