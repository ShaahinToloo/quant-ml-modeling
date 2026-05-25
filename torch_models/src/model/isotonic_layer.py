import torch
import torch.nn as nn

class IsotonicLayer(nn.Module):
    def __init__(self, x_points, y_points):
        super().__init__()
        self.register_buffer("x_points", torch.tensor(x_points, dtype=torch.float32))
        self.register_buffer("y_points", torch.tensor(y_points, dtype=torch.float32))

    def forward(self, x):
        x = torch.clamp(x, self.x_points[0], self.x_points[-1])

        idx = torch.bucketize(x, self.x_points) - 1
        idx = torch.clamp(idx, 0, len(self.x_points)-2)

        x0 = self.x_points[idx]
        x1 = self.x_points[idx+1]
        y0 = self.y_points[idx]
        y1 = self.y_points[idx+1]

        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

