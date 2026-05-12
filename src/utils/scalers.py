import torch


class Angle_Scaler:
    def __init__(self, min_angle: float, max_angle: float):
        self.min_angle = min_angle  #'degree'
        self.max_angle = max_angle  #'degree'

    def scale(self, angles):
        return (angles - self.min_angle) / (self.max_angle - self.min_angle)

    def rescale(self, scaled_batch_angle):
        return (
            torch.clip(scaled_batch_angle, min=0.0, max=1.0)
            * (self.max_angle - self.min_angle)
            + self.min_angle
        )


class ABC_Scaler:
    def __init__(
        self, init_batch_abc, min_length: float, max_length: float, device="cuda"
    ):
        self.init_batch_abc = init_batch_abc
        self.base_length = (
            (torch.max(self.init_batch_abc, dim=1).values).view(-1, 1).to(device)
        )
        self.min_length = min_length
        self.max_length = max_length
        self.device = device
        ## あとで消す
        # self.base_length = self.max_length

    def update_base_length(self):
        self.base_length = (
            (torch.max(self.init_batch_abc, dim=1).values).view(-1, 1).to(self.device)
        )
        ## あとで消す
        # self.base_length = self.max_length

    def scale(self, batch_abc):
        return batch_abc / self.base_length

    def rescale(self, scaled_batch_abc):
        return torch.clip(
            scaled_batch_abc * self.base_length,
            min=self.min_length,
            max=self.max_length,
        )
