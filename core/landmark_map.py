from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

GlobalPt = Tuple[float, float]


@dataclass
class Landmark:
    """
    最简地标结构
    id: 稳定的 Landmark ID（全局去重计数的核心）
    pos_g: (x, y) in global/world frame
    state: "active" or "picked"
    last_seen_t: 最近一次被观测/更新的时间（秒）
    """
    id: int             # 唯一坐标ID
    pos_g: GlobalPt     # 全局坐标
    state: str          # "active" or "picked"
    last_seen_t: float  # seconds


@dataclass(frozen=True)
class Observation:
    """一次观测（在全局坐标系下）。"""
    pos_g: GlobalPt
    conf: float = 1.0   # 置信度
    t: float = 0.0      # 0 表示用当前时间

# 定义一个二维欧式距离平方函数：之后用于计算 距离门限的最近邻关联
def _dist2(a: GlobalPt, b: GlobalPt) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


class LandmarkMap:
    """
    全 Landmark 管理：
      - update(): 数据关联+新建
      - mark_picked(): 拾取闭环更新
      - choose_next_target(): 贪心选最近 active
    仅基于几何一致性（距离门限），不做外观 ReID。
    """

    def __init__(
        self,
        assoc_dist: float = 0.25,   # 距离门限：米
        pos_ema_alpha: float = 0.7, # 地标位置更新 的指数滑动平均系数，控制“稳定性 vs 响应速度” 位置 EMA: new = a*old + (1-a)*obs
        init_min_conf: float = 0.3, # 新建地标最小 conf
        max_landmarks: int = 500,
        stale_time: float = 10.0,   # 降权时间：可用于清理
    ) -> None:
        self.assoc_dist = float(assoc_dist)
        self.assoc_dist2 = self.assoc_dist * self.assoc_dist
        self.pos_ema_alpha = float(pos_ema_alpha)
        self.init_min_conf = float(init_min_conf)
        self.max_landmarks = int(max_landmarks)
        self.stale_time = float(stale_time)

        self._next_id = 1
        self._lms: Dict[int, Landmark] = {}

    # ---------- 查询 ----------
    def all(self) -> List[Landmark]:
        return list(self._lms.values())

    def active(self) -> List[Landmark]:
        return [lm for lm in self._lms.values() if lm.state == "active"]

    def picked(self) -> List[Landmark]:
        return [lm for lm in self._lms.values() if lm.state == "picked"]

    def get(self, lid: int) -> Optional[Landmark]:
        return self._lms.get(lid)
    #全局去重后的地标总数
    def count_total(self) -> int:
        return len(self._lms)

    def count_active(self) -> int:
        return sum(1 for lm in self._lms.values() if lm.state == "active")

    def count_picked(self) -> int:
        return sum(1 for lm in self._lms.values() if lm.state == "picked")

    # ---------- 核心：更新 ----------
    def update(self, obs_list: List[Observation], now_t: Optional[float] = None) -> List[Tuple[int, Observation]]:
        """
        输入：一批全局观测点
        输出：[(landmark_id, obs), ...]
        """
        # 更新时间，若没有就用电脑时间
        if now_t is None:
            now_t = time.time()

        if len(self._lms) > self.max_landmarks:
            self._prune(now_t)

        out: List[Tuple[int, Observation]] = []

        for obs in obs_list:
            t = obs.t if obs.t > 0 else now_t
            obs2 = Observation(pos_g=obs.pos_g, conf=obs.conf, t=t)

            # 置信度太低：不新建
            if obs2.conf < self.init_min_conf:
                continue

            lid = self._associate_nearest(obs2.pos_g)

            if lid is None:
                lid = self._create(obs2)
            else:
                self._update_pos(lid, obs2)

            out.append((lid, obs2))

        return out

    # 用于新旧球之间的关联
    def _associate_nearest(self, p: GlobalPt) -> Optional[int]:#内部使用的私有方法
        """在 active 地标中做最近邻 + 距离门限 gating。"""
        best_id = None
        best_d2 = float("inf")#初始设置为正无穷大

        for lm in self._lms.values():#在landmark中取出所有进行遍历
            if lm.state != "active":
                continue
            d2 = _dist2(p, lm.pos_g)
            if d2 < best_d2:#比较欧式距离平方
                best_d2 = d2
                best_id = lm.id

        if best_id is None:
            return None
        return best_id if best_d2 <= self.assoc_dist2 else None

    def _create(self, obs: Observation) -> int:
        lid = self._next_id
        self._next_id += 1
        self._lms[lid] = Landmark(
            id=lid,
            pos_g=obs.pos_g,
            state="active",
            last_seen_t=obs.t,
        )
        return lid

    def _update_pos(self, lid: int, obs: Observation) -> None:
        lm = self._lms[lid]     # 后续修改直接作用到地图当中
        a = self.pos_ema_alpha  # 历史权重
        # 元组解宝（Tuple Unpacking）
        x_old, y_old = lm.pos_g
        x_obs, y_obs = obs.pos_g
        # 指数滑动平均（EMA）
        lm.pos_g = (a * x_old + (1 - a) * x_obs, a * y_old + (1 - a) * y_obs)
        lm.last_seen_t = obs.t  # 更新该地标最新一次的观测时间

    # ---------- 拾取闭环 ----------
    # 指定 landmark 标记为“已拾取”，并更新时间戳
    def mark_picked(self, lid: int, picked_t: Optional[float] = None) -> bool:
        lm = self._lms.get(lid)
        if lm is None:
            return False
        lm.state = "picked"
        lm.last_seen_t = picked_t if picked_t is not None else time.time()
        return True

    # 基于我们的铲车/吸入式机器
    def mark_picked_region(self, pick_center_g: GlobalPt, pick_radius: float, picked_t: Optional[float] = None) -> List[int]:
        t = picked_t if picked_t is not None else time.time()
        picked_ids = []
        r2 = pick_radius * pick_radius
        for lm in self._lms.values():
            if lm.state != "active":
                continue
            if _dist2(lm.pos_g, pick_center_g) <= r2:
                lm.state = "picked"
                lm.last_seen_t = t
                picked_ids.append(lm.id)
        return picked_ids

    def choose_next_target(self, robot_pos_g: GlobalPt) -> Optional[int]:
        """贪心：选离机器人最近的 active 地标。"""
        best_id = None
        best_d2 = float("inf")

        for lm in self._lms.values():
            if lm.state != "active":
                continue
            d2 = _dist2(robot_pos_g, lm.pos_g)
            if d2 < best_d2:
                best_d2 = d2
                best_id = lm.id

        return best_id

    def target_reached(self, robot_pos_g: GlobalPt, lid: int, reach_dist: float = 0.20) -> bool:
        # reach_distinct判定半径，需要修改
        lm = self._lms.get(lid)
        if lm is None or lm.state != "active":
            return False
        return _dist2(robot_pos_g, lm.pos_g) <= reach_dist * reach_dist

    # ---------- 可选：清理 ----------
    def _prune(self, now_t: float) -> None:
        """
        简单清理：删掉“很久没见过的 active 地标”（通常是噪声或已离开视野的误检）。
        注意：这会改变“计数”含义（删除后总数会变少）。
        如果论文里要“全局累计计数”，建议不要删除，而是保留并用 state/age 管理。
        """
        to_del = []
        for lm in self._lms.values():
            if lm.state != "active":
                continue
            if (now_t - lm.last_seen_t) > self.stale_time:
                to_del.append(lm.id)

        # 每次删一部分，避免突然大量删除
        for lid in to_del[: max(10, len(to_del) // 5)]:
            self._lms.pop(lid, None)

    # 热力图方法
    def choose_next_target_with_heatmap(
            self,
            robot_pos_g,
            heatmap,
            min_landmarks: int = 3,
    ):
        active = self.active()
        if len(active) >= min_landmarks:
            return self.choose_next_target(robot_pos_g), "landmark"
        else:
            return heatmap.get_peak(), "heatmap"

if __name__ == "__main__":
    m = LandmarkMap(assoc_dist=0.30)

    assoc = m.update([
        Observation((1.0, 2.0), conf=0.9),
        Observation((1.1, 2.05), conf=0.8),  # 应关联到同一个 lid
        Observation((5.0, 1.0), conf=0.95),
    ])

    print("assoc:", assoc)
    print("total:", m.count_total(), "active:", m.count_active())

    robot = (0.0, 0.0)
    target = m.choose_next_target(robot)
    if target is not None and m.target_reached(robot, target):
        pick_center = robot  # 或者用“铲口点”更准确
        r_pick = 0.35  # 拾取半径（按机构实测）
        picked_ids = m.mark_picked_region(pick_center, r_pick)
        print("picked:", picked_ids)  # 一次可能拾取多个

