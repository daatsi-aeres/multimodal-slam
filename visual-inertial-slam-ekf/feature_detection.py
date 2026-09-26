import numpy as np
import cv2


def detect_features_stereo(img_left, img_right, max_corners=500,
                           quality_level=0.01, min_distance=10):
    if img_left.dtype != np.uint8:
        img_left = (img_left * 255).astype(np.uint8) if img_left.max() <= 1.0 else img_left.astype(np.uint8)
    if img_right.dtype != np.uint8:
        img_right = (img_right * 255).astype(np.uint8) if img_right.max() <= 1.0 else img_right.astype(np.uint8)

    corners = cv2.goodFeaturesToTrack(
        img_left, maxCorners=max_corners,
        qualityLevel=quality_level, minDistance=min_distance
    )
    if corners is None or len(corners) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    pts_left = corners.reshape(-1, 2).astype(np.float32)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005)
    )

    pts_right, status, _ = cv2.calcOpticalFlowPyrLK(
        img_left, img_right, pts_left, None, **lk_params
    )
    status = status.flatten().astype(bool)

    if pts_right is not None:
        y_diff = np.abs(pts_left[:, 1] - pts_right[:, 1])
        epipolar_ok = y_diff < 1.0
        valid = status & epipolar_ok
    else:
        valid = status

    return pts_left[valid], pts_right[valid]


def track_features_temporal(prev_img, curr_img, prev_pts):
    if len(prev_pts) == 0:
        return prev_pts.copy(), np.array([], dtype=bool)

    for img in [prev_img, curr_img]:
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

    prev_pts_f = prev_pts.astype(np.float32)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005)
    )

    curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_img, curr_img, prev_pts_f, None, **lk_params
    )

    if curr_pts is not None:
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_img, prev_img, curr_pts, None, **lk_params
        )
        fb_error = np.linalg.norm(prev_pts_f - back_pts, axis=1)
        valid = (status_fwd.flatten().astype(bool) &
                 status_bwd.flatten().astype(bool) &
                 (fb_error < 2.0))
    else:
        valid = np.zeros(len(prev_pts), dtype=bool)

    tracked = np.full_like(prev_pts, np.nan)
    if curr_pts is not None:
        tracked[valid] = curr_pts[valid]

    return tracked, valid


def build_feature_matrix(vid_l_path, vid_r_path, n_timesteps, max_corners=150,
                         quality_level=0.05, min_distance=30,
                         detect_interval=20, verbose=True):
    cap_l = cv2.VideoCapture(vid_l_path)
    cap_r = cv2.VideoCapture(vid_r_path)

    tracks = []
    next_id = 0
    active_pts = np.empty((0, 2), dtype=np.float32)
    active_ids = []

    lk_params = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005)
    )

    prev_img_l = None

    for t in range(n_timesteps):
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        img_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        if verbose and t % 100 == 0:
            print(f"  Processing frame {t}/{n_timesteps} "
                  f"(active: {len(active_ids)}, total tracks: {len(tracks)})")

        if t > 0 and len(active_pts) > 0 and prev_img_l is not None:
            curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                prev_img_l, img_l, active_pts.astype(np.float32), None, **lk_params
            )
            if curr_pts is not None:
                valid = status_fwd.flatten().astype(bool)
            else:
                valid = np.zeros(len(active_pts), dtype=bool)

            new_active_pts = []
            new_active_ids = []
            for i, (aid, v) in enumerate(zip(active_ids, valid)):
                if v and curr_pts is not None:
                    new_active_pts.append(curr_pts[i])
                    new_active_ids.append(aid)

            active_pts = np.array(new_active_pts, dtype=np.float32).reshape(-1, 2) if new_active_pts else np.empty((0, 2), dtype=np.float32)
            active_ids = new_active_ids

        if t % detect_interval == 0 or len(active_ids) < max_corners // 2:
            H, W = img_l.shape[:2]
            mask = np.ones((H, W), dtype=np.uint8) * 255
            for pt in active_pts:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(mask, (x, y), int(min_distance), 0, -1)

            new_corners = cv2.goodFeaturesToTrack(
                img_l, maxCorners=max(max_corners - len(active_ids), 50),
                qualityLevel=quality_level, minDistance=min_distance, mask=mask
            )
            if new_corners is not None and len(new_corners) > 0:
                new_pts = new_corners.reshape(-1, 2).astype(np.float32)
                for pt in new_pts:
                    tracks.append({'id': next_id, 'observations': {}})
                    active_ids.append(next_id)
                    next_id += 1
                active_pts = np.vstack([active_pts, new_pts]) if len(active_pts) > 0 else new_pts

        if len(active_pts) > 0:
            right_pts, status_stereo, _ = cv2.calcOpticalFlowPyrLK(
                img_l, img_r, active_pts.astype(np.float32), None, **lk_params
            )
            status_stereo = status_stereo.flatten().astype(bool) if status_stereo is not None else np.zeros(len(active_pts), dtype=bool)

            if right_pts is not None:
                y_diff = np.abs(active_pts[:, 1] - right_pts[:, 1])
                stereo_valid = status_stereo & (y_diff < 1.0)
            else:
                stereo_valid = np.zeros(len(active_pts), dtype=bool)

            for i, aid in enumerate(active_ids):
                track = tracks[aid]
                if stereo_valid[i]:
                    track['observations'][t] = (
                        active_pts[i, 0], active_pts[i, 1],
                        right_pts[i, 0], right_pts[i, 1]
                    )

        prev_img_l = img_l.copy()

    cap_l.release()
    cap_r.release()

    M = len(tracks)
    features = np.full((4, M, n_timesteps), -1.0, dtype=np.float32)

    for track in tracks:
        tid = track['id']
        for t, (uL, vL, uR, vR) in track['observations'].items():
            features[0, tid, t] = uL
            features[1, tid, t] = vL
            features[2, tid, t] = uR
            features[3, tid, t] = vR

    if verbose:
        n_valid = np.sum(np.all(features != -1, axis=0))
        print(f"  Built feature matrix: shape={features.shape}, "
              f"total valid observations={n_valid}")

    return features
