import argparse
import os
import numpy as np
import glob
import cv2
import tqdm
import ffmpeg


def get_frame_shape(npz_dir):
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No .npz files found in the specified directory: {npz_dir}")

    sample_data = np.load(npz_files[0])
    if 'x' in sample_data and 'y' in sample_data:
        height = sample_data['y'].max() + 1
        width = sample_data['x'].max() + 1
    else:
        raise ValueError("The .npz file does not contain 'x' and 'y' arrays for determining frame shape.")

    return height, width


def write_frames_to_video(output_video, framerate=60, vcodec='libx264'):
    npz_dir = "~/event_camera/sanky/rpg_vid2e/acino_test/sreepranav_events_final_true1"
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in directory: {npz_dir}")
        return

    print(f"Detected {len(npz_files)} .npz files in directory: {npz_dir}")
    height, width = get_frame_shape(npz_dir)
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
    args = parser.parse_args()

    write_frames_to_video(args.npz_dir, args.output_video, framerate=args.framerate)
