import numpy as np


vis_mode = False             # 用于可视化debug
def get_motion_transform(ego_v, ego_yawrate, dt):
    """
    计算从上一帧坐标系到当前帧坐标系的变换矩阵 (Ego Motion Compensation)
    """
    dx = ego_v * dt
    dtheta = ego_yawrate * dt
    transform = np.array([
        [np.cos(dtheta),  np.sin(dtheta), -dx * np.cos(dtheta)],
        [-np.sin(dtheta), np.cos(dtheta),  dx * np.sin(dtheta)],
        [0, 0, 1]
    ])
    return transform


def transform_points(points, transform_matrix):
    """
    points: (N, 2)
    transform_matrix: (3, 3)
    return: (N, 2) 转换到当前帧的点
    """
    if len(points) == 0:
        return points
    points_h = np.c_[points, np.ones(len(points))]
    # (3, 3) @ (3, N) -> (3, N) -> .T -> (N, 3)
    points_curr_h = (transform_matrix @ points_h.T).T
    return points_curr_h[:, :2]


def get_dist_cost(boxes_a, boxes_b):
    """计算欧式距离"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)))
    # boxes shape: (N, 2)
    diff = boxes_a[:, np.newaxis, :] - boxes_b[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return dist


def greedy_linear_assignment(cost_matrix):
    """贪婪匹配"""
    if cost_matrix.size == 0:
        return np.array([]), np.array([])
    cost = cost_matrix.copy()
    rows, cols = cost.shape
    # 记录匹配结果
    row_ind = []
    col_ind = []
    # 矩阵排序，优先处理最小代价
    sorted_indices = np.argsort(cost, axis=None)
    # 使用 mask 标记已匹配的行和列
    row_covered = np.zeros(rows, dtype=bool)
    col_covered = np.zeros(cols, dtype=bool)
    for idx in sorted_indices:
        r = idx // cols
        c = idx % cols
        if row_covered[r] or col_covered[c]:
            continue
        # 记录匹配
        row_ind.append(r)
        col_ind.append(c)
        # 标记
        row_covered[r] = True
        col_covered[c] = True
        # 提前终止：如果所有行或所有列都已匹配
        if len(row_ind) >= min(rows, cols):
            break
    return np.array(row_ind), np.array(col_ind)


def get_system_aeb_level(cam_output):
    """取所有障碍物AEB_level的最大值，返回系统级AEB指令整数0-5。"""
    levels = cam_output.get('AEB_level', [])
    return int(max(levels)) if levels else 0


def cal_cam_output(bbox_infos,
                   ego_vels,
                   ego_yawrates,
                   det_thresh=0.4,
                   track_len=3,
                   DT=0.5,
                   max_connect_dist=3.0,
                   front_thresh=3.5,
                   ttc_w1=4.0,
                   ttc_w2=3.0,
                   ttc_b1=2.0,
                   ttc_b2=1.5,
                   ttc_b3=1.0,
                   lateral_zone_y=5.0,
                   lateral_vel_thresh=1.5,
                   lateral_threat_level=1):
    """
    bbox_infos: 最近3帧的 bbox 信息，idx 2 为当前帧
    ego_vels:   最近3帧ego速度信息，idx 2 为当前帧
    ego_yawrates: 最近3帧ego偏航角速度信息，idx 2 为当前帧
    track_len: 计算mot速度的时序帧数
    DT: 帧间隔时间
    max_connect_dist: 轨迹匹配的最大距离阈值
    front_thresh: 纵向TTC计算的横向范围（m）
    ttc_w1/w2: 预警1/2的TTC阈值（s）
    ttc_b1/b2/b3: 刹车1/2/3的TTC阈值（s）
    lateral_zone_y: 横向cut-in监控半宽（m）
    lateral_vel_thresh: 触发cut-in的横向相对速度阈值（m/s）
    lateral_threat_level: cut-in威胁对应的AEB档位（1=W1）
    """
    cam_output = {}
    # 1. 数据预处理：提取 XY 和 Box
    frames_dets = []
    for t in range(track_len):
        frame_data = bbox_infos[t]
        valid_mask = np.array(frame_data['scores_3d']) > det_thresh
        try:
            boxes = np.array(frame_data['boxes_3d'].tensor)[valid_mask]
        except:
            boxes = np.array(frame_data['boxes_3d'])[valid_mask]
        types = np.array(frame_data['labels_3d'])[valid_mask]
        dets = {'boxes': boxes, 'xy': boxes[:, :2], 'types': types}
        frames_dets.append(dets)

    # 2. Forward Tracking
    # active_tracks 结构: [{'history': [(frame_idx, pos_xy), ...]}, ...]
    active_tracks = []
    for t in range(track_len):
        dets_xy = frames_dets[t]['xy']
        # 第一帧，直接初始化轨迹
        if t == 0:
            for i in range(len(dets_xy)):
                active_tracks.append({'history': [(t, dets_xy[i])]})
            continue
        # 计算从 t-1 到 t 的变换矩阵
        T_prev_to_curr = get_motion_transform(ego_vels[t], ego_yawrates[t], DT)
        pred_positions = []
        track_indices = []  # 记录参与匹配的 active_track 索引
        for idx, track in enumerate(active_tracks):
            last_t, last_pos = track['history'][-1]
            # 只有上一帧存在的轨迹才参与预测匹配
            if last_t == t - 1:
                # 变换点到当前帧坐标系
                pred_pos = transform_points(last_pos[np.newaxis, :], T_prev_to_curr)[0]
                pred_positions.append(pred_pos)
                track_indices.append(idx)
        pred_positions = np.array(pred_positions)
        # 贪心匹配
        cost_matrix = get_dist_cost(pred_positions, dets_xy)
        matched_track_indices = set()
        matched_det_indices = set()
        if cost_matrix.size > 0:
            row_ind, col_ind = greedy_linear_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_connect_dist: # 距离阈值 3.0m
                    real_track_idx = track_indices[r]
                    active_tracks[real_track_idx]['history'].append((t, dets_xy[c]))
                    matched_track_indices.add(real_track_idx)
                    matched_det_indices.add(c)
        for i in range(len(dets_xy)):
            if i not in matched_det_indices:
                active_tracks.append({'history': [(t, dets_xy[i])]})

    # 3. 速度计算
    final_output_list = []
    cur_t = track_len - 1
    current_frame_dets = frames_dets[cur_t]
    for i in range(len(current_frame_dets['boxes'])):
        aeb_level = 0
        bbox = current_frame_dets['boxes'][i]
        type = current_frame_dets['types'][i]
        center = current_frame_dets['xy'][i]
        # 找到与当前检测框对应的 track
        matched_track = None
        # 倒序查找
        for track in reversed(active_tracks):
            last_t, last_pos = track['history'][-1]
            # 必须匹配到当前帧
            if last_t == cur_t and np.linalg.norm(last_pos - center) < 0.01:
                matched_track = track
                break
        rel_vel_x = 0.0
        rel_vel_y = 0.0
        ttc = 99.0
        # 有历史才计算速度
        if matched_track and len(matched_track['history']) >= 2:
            compensated_history = []
            # 将历史点全部变换到当前帧的坐标系下
            for (hist_t, hist_pos) in matched_track['history']:
                p_trans = hist_pos.copy()
                # 累积变换: 从 hist_t+1 变换到 cur_t
                if hist_t < cur_t:
                    for k in range(hist_t + 1, cur_t + 1):
                        T = get_motion_transform(ego_vels[k], ego_yawrates[k], DT)
                        p_trans = (T @ np.append(p_trans, 1.0))[:2]
                compensated_history.append(p_trans)
            compensated_history = np.array(compensated_history)

            # 线性拟合计算速度
            times = (np.array([h[0] for h in matched_track['history']]) - cur_t) * DT
            # 当前帧自车速度
            curr_ego_v = ego_vels[cur_t]
            if len(times) > 1:
                xs = compensated_history[:, 0]
                ys = compensated_history[:, 1]
                # 最小二乘计算速度：v = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(xx) - sum(x)^2)
                n = len(times)
                sum_t = np.sum(times)
                sum_t_sq = np.sum(times**2)
                denominator = n * sum_t_sq - sum_t**2
                if abs(denominator) < 1e-6:
                    # 所有历史点时间戳相同，无法估计速度，保持默认安全值
                    pass
                else:
                    # 计算绝对速度
                    abs_vel_x = (n * np.sum(times * xs) - sum_t * np.sum(xs)) / denominator
                    abs_vel_y = (n * np.sum(times * ys) - sum_t * np.sum(ys)) / denominator
                    # 计算相对速度
                    rel_vel_x = abs_vel_x - curr_ego_v
                    rel_vel_y = abs_vel_y  # 自车横向速度忽略不计

                    # 纵向TTC与5档AEB分类
                    if bbox[0] > 0 and abs(bbox[1]) < front_thresh and rel_vel_x < 0:
                        # TTC = 距离 / 相对接近速度（rel_vel_x < 0 保证TTC为正）
                        ttc = bbox[0] / (-rel_vel_x)
                        if ttc <= ttc_b3:
                            aeb_level = 5   # B3: 紧急全刹
                        elif ttc <= ttc_b2:
                            aeb_level = 4   # B2: 中度刹车
                        elif ttc <= ttc_b1:
                            aeb_level = 3   # B1: 部分刹车
                        elif ttc <= ttc_w2:
                            aeb_level = 2   # W2: 强提醒
                        elif ttc <= ttc_w1:
                            aeb_level = 1   # W1: 软提醒

                    # 横向cut-in威胁检测
                    if (bbox[0] > 0
                            and abs(bbox[1]) >= front_thresh
                            and abs(bbox[1]) < lateral_zone_y
                            and abs(rel_vel_y) > lateral_vel_thresh
                            and np.sign(rel_vel_y) == -np.sign(bbox[1])):
                        aeb_level = max(aeb_level, lateral_threat_level)

        # 向后兼容别名
        main_obstacle = aeb_level >= 1
        dangerous_obstacle = aeb_level >= 5

        obj_data = {
            'Obstacle_Pos_X': float(bbox[0]),
            'Obstacle_Pos_Y': float(bbox[1]),
            'Obstacle_Width': float(bbox[4]),
            'Obstacle_Height': float(bbox[5]),
            'Obstacle_Rel_Vel_X': float(rel_vel_x),
            'Obstacle_Rel_Vel_Y': float(rel_vel_y),
            'Obstacle_TTC': float(ttc),
            'Obstacle_type': type,
            'AEB_level': int(aeb_level),
            'Main_obstacle': main_obstacle,
            'Dangerous_obstacle': dangerous_obstacle,
        }

        if vis_mode:
            # 用于可视化debug
            obj_data['Compensated_History'] = compensated_history if 'compensated_history' in locals() else []
        final_output_list.append(obj_data)
    # final_output_list.sort(key=lambda x: x['Obstacle_TTC'])
    keys = ['Obstacle_Pos_X', 'Obstacle_Pos_Y', 'Obstacle_Rel_Vel_X',
            'Obstacle_Rel_Vel_Y', 'Obstacle_type', 'Obstacle_Width',
            'Obstacle_Height', 'Obstacle_TTC',
            'AEB_level', 'Main_obstacle', 'Dangerous_obstacle']
    if vis_mode:
        keys.append('Compensated_History')
    for k in keys:
        cam_output[k] = [item[k] for item in final_output_list]

    return cam_output
