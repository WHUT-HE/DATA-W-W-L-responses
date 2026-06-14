import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc  # ✅ 替代 pyDOE2

# ==== STEP 1: 加载已有数据 ====
# 请替换下面两行，使用你自己的风浪响应 & 泥流响应时程数据
# 示例数据加载方式：一列 txt 文件，长度与时间轴一致
dt = 0.1
t = np.arange(0, 35.1, dt)

# -----------------------
theta_windwave = np.loadtxt("theta_windwave.txt")           # 风浪响应转角时程
theta_mudflow_unit = np.loadtxt("滑坡-h4-v8.txt")   # 泥流响应（从 t=0 开始）
# -----------------------


assert len(theta_windwave) == len(t), "风浪响应长度应等于时间向量长度"
assert len(theta_mudflow_unit) * dt <= t[-1], "泥流响应时间长度超出总时长"

# ==== STEP 2: 拉丁超立方采样泥流发生时刻 ====
n_samples = 500
mudflow_time_range = (0, 28.5)

sampler = qmc.LatinHypercube(d=1)
samples = sampler.random(n=n_samples)
t_mud_samples = mudflow_time_range[0] + (mudflow_time_range[1] - mudflow_time_range[0]) * samples.flatten()

# ==== STEP 3: 逐样本叠加响应，判定失效 ====
theta_limit = 0.5
fail_count = 0
theta_all = []

n_mud = len(theta_mudflow_unit)

for t_mud in t_mud_samples:
    start_idx = int(t_mud / dt)
    theta_total = theta_windwave.copy()

    if start_idx + n_mud <= len(t):
        theta_total[start_idx:start_idx + n_mud] += theta_mudflow_unit
    else:
        n_valid = len(t) - start_idx
        theta_total[start_idx:] += theta_mudflow_unit[:n_valid]

    theta_all.append(theta_total)

    if np.max(theta_total) > theta_limit:
        fail_count += 1

# ==== STEP 4: 输出失效概率 ====
P_failure = fail_count / n_samples
print(f"在 {n_samples} 个ITOL的海底滑坡样本中，海上风机结构转角超过 {theta_limit:.4f} rad 的概率为：{P_failure:.2%}")

# ==== STEP 5: 可视化一个样本 ====
plt.figure(figsize=(10, 6))
plt.plot(t, theta_windwave, label='wind-waveresponse', color='blue')
plt.plot(t, theta_all[0], label='wind-wave-debris', color='red', alpha=0.7)
plt.axhline(theta_limit, color='black', linestyle='--', label='lim')
plt.xlabel("time (s)")
plt.ylabel("rotation (rad)")
plt.title("failure analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
