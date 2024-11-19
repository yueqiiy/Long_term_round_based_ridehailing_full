from typing import Dict, Any


class Pair:
    def __init__(self, vehicle_id: int,order_id: int, weight: float):
        self.vehicle_id = vehicle_id
        self.order_id = order_id
        self.od_distance: float
        # self.pick_up_eta:
        self.weight = weight
        self.duration: float

    def redefine_weight(self, score):
        self.weight = score