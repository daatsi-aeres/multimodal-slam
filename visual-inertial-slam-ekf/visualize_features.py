import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_detection import detect_features_stereo


def visualize_stereo_matches(img_left, img_right, pts_left, pts_right,
                             frame_idx, save_path=None):
    if len(img_left.shape) == 2:
        img_left_c = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    else:
        img_left_c = img_left.copy()
    if len(img_right.shape) == 2:
        img_right_c = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    else:
        img_right_c = img_right.copy()

    H, W = img_left_c.shape[:2]

    vis = np.hstack([img_left_c, img_right_c])

    n_matches = len(pts_left)
    colors = []
    np.random.seed(frame_idx)
    for i in range(n_matches):
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
        colors.append(color)

    for i in range(n_matches):
        pt_l = (int(pts_left[i, 0]), int(pts_left[i, 1]))
        pt_r = (int(pts_right[i, 0]) + W, int(pts_right[i, 1]))
        cv2.line(vis, pt_l, pt_r, (0, 200, 0), 1, cv2.LINE_AA)

    for i in range(n_matches):
        pt_l = (int(pts_left[i, 0]), int(pts_left[i, 1]))
        pt_r = (int(pts_right[i, 0]) + W, int(pts_right[i, 1]))
        cv2.circle(vis, pt_l, 4, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, pt_r, 4, (0, 0, 255), -1, cv2.LINE_AA)

    title = f"Frame {frame_idx}  |  {n_matches} stereo matches  |  Left (blue) / Right (red)"
    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(vis, "LEFT", (W // 2 - 30, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(vis, "RIGHT", (W + W // 2 - 40, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
        print(f"  [SAVED] {save_path}")

    return vis


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset02')
    data_dir = os.path.abspath(data_dir)
    vid_l_path = os.path.join(data_dir, 'dataset02_l.mp4')
    vid_r_path = os.path.join(data_dir, 'dataset02_r.mp4')

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results',
                               'dataset02_extra_credit', 'plots')
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    cap_l = cv2.VideoCapture(vid_l_path)
    cap_r = cv2.VideoCapture(vid_r_path)

    total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    sample_frames = np.linspace(50, total_frames - 50, 5, dtype=int)
    print(f"Sampling frames: {sample_frames}")

    print(f"\n{'='*60}")
    print(f" Feature Detection Visualization — Dataset 02")
    print(f"{'='*60}\n")

    for frame_idx in sample_frames:
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print(f"  Could not read frame {frame_idx}, skipping")
            continue

        img_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        pts_left, pts_right = detect_features_stereo(
            img_l, img_r, max_corners=100,
            quality_level=0.05, min_distance=30
        )

        print(f"  Frame {frame_idx}: {len(pts_left)} stereo matches")

        save_path = os.path.join(results_dir, f'feature_match_frame{frame_idx:04d}.png')
        visualize_stereo_matches(
            img_l, img_r, pts_left, pts_right,
            frame_idx, save_path
        )

    cap_l.release()
    cap_r.release()

    print(f"\n{'='*60}")
    print(f" DONE — Visualizations saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
