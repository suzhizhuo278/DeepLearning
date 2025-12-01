import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =========================
# 1. 真实动力系统：螺旋 ODE
# =========================
# dz/dt = A z
A = torch.tensor([[-0.1, -1.0],
                  [ 1.0, -0.1]], dtype=torch.float32)


def true_dynamics(z):
    """
    z: (..., 2)
    返回 dz/dt = A z
    """
    return z @ A.T  # (..., 2)


# =========================
# 2. 手写 ODE 求解器：RK4
# =========================
def rk4_step(f, z, t, dt):
    """
    单步 RK4：
    z_{n+1} = z_n + dt/6 (k1 + 2k2 + 2k3 + k4)
    这里 f(z, t) = dz/dt
    """
    k1 = f(z, t)
    k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(z + dt * k3, t + dt)
    return z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def ode_solve(f, z0, t_grid):
    """
    用 RK4 在给定时间网格 t_grid 上求解 ODE。
    z0: (batch, dim)
    t_grid: (T,) 升序
    返回：z_traj (T, batch, dim)
    """
    zs = [z0]
    z = z0
    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        dt = t_grid[i+1] - t
        z = rk4_step(f, z, t, dt)
        zs.append(z)
    return torch.stack(zs, dim=0)  # (T, batch, dim)


# =========================
# 3. 生成训练数据（用真实 ODE）
# =========================
def generate_data(num_traj=64, T=5.0, steps=100):
    """
    生成 num_traj 条轨迹，每条从随机初始点出发。
    """
    t_grid = torch.linspace(0., T, steps)
    # 随机初始点，均匀在 [-2, 2]^2
    z0 = torch.empty(num_traj, 2).uniform_(-2., 2.)
    with torch.no_grad():
        z_traj = ode_solve(lambda z, t: true_dynamics(z), z0, t_grid)
    # z_traj: (steps, num_traj, 2)
    return t_grid, z0, z_traj


# =========================
# 4. Neural ODE 模型：f_theta
# =========================
class NeuralDynamics(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        # 简单 MLP：输入 (z, t) → 输出 dz/dt
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, t):
        """
        z: (batch, dim)
        t: float 或标量张量
        返回 dz/dt: (batch, dim)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=z.dtype, device=z.device)
        # 把 t 扩展到和 batch 一样多
        t_expand = t.expand(z.shape[0], 1)
        inp = torch.cat([z, t_expand], dim=-1)
        return self.net(inp)


# =========================
# 5. 训练 Neural ODE
# =========================
def train_neural_ode(num_epochs=2000, lr=1e-3, device="cpu"):
    # 数据
    t_grid, z0, z_traj = generate_data()
    t_grid = t_grid.to(device)
    z0 = z0.to(device)
    z_traj = z_traj.to(device)

    model = NeuralDynamics(dim=2, hidden=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def f_theta(z, t):
        return model(z, t)

    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()
        # 用 Neural ODE 从 z0 推出整条轨迹
        z_pred = ode_solve(f_theta, z0, t_grid)  # (steps, batch, 2)
        loss = loss_fn(z_pred, z_traj)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, loss = {loss.item():.6f}")

    return model, (t_grid, z0, z_traj)


# =========================
# 6. 训练并可视化
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, (t_grid, z0, z_traj_true) = train_neural_ode(device=device)

    # 用训练好的模型从新初始点出发生成轨迹，对比真实系统
    with torch.no_grad():
        t_grid = t_grid.to(device)
        z0_test = torch.tensor([[1.5, 0.0],
                                [0.0, 2.0],
                                [-1.5, -1.5]], dtype=torch.float32).to(device)

        z_traj_true_test = ode_solve(lambda z, t: true_dynamics(z), z0_test, t_grid)
        z_traj_pred_test = ode_solve(lambda z, t: model(z, t), z0_test, t_grid)

    z_traj_true_test = z_traj_true_test.cpu().numpy()  # (steps, 3, 2)
    z_traj_pred_test = z_traj_pred_test.cpu().numpy()

    # 画出真实轨迹 vs Neural ODE 轨迹
    plt.figure(figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i in range(z0_test.shape[0]):
        plt.plot(z_traj_true_test[:, i, 0],
                 z_traj_true_test[:, i, 1],
                 linestyle="--", label=f"true traj {i+1}", alpha=0.7)
        plt.plot(z_traj_pred_test[:, i, 0],
                 z_traj_pred_test[:, i, 1],
                 linestyle="-", label=f"neural ODE {i+1}", alpha=0.7)

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.legend()
    plt.title("True dynamics vs Neural ODE learned dynamics")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()