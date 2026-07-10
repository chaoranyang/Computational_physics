# 交通流方程与von Neumann 稳定性分析

https://github.com/user-attachments/assets/ab7811a5-c291-4dea-8ca2-ba381e148c56

<video autoplay loop muted playsinline style="max-width: 100%; height: auto;">
  <source src="https://github.com/user-attachments/assets/ab7811a5-c291-4dea-8ca2-ba381e148c56" type="video/mp4">
</video>
---

## 1. 物理背景与模型方程

交通流问题中，车辆密度 $\rho(x,t)$ 满足连续性方程。若将流量简化为 $F = C \rho$（ $C$ 为常数波速），则得到线性平流方程

$$
\frac{\partial \rho}{\partial t} = -C \frac{\partial \rho}{\partial x}
$$

该方程描述密度波以速度 $C$ 沿 $x$ 方向传播。本任务取空间区间 $L = 400$，波速 $C = 1$，使用 Lax‑Wendroff 格式进行数值模拟，观察不同初始条件下的密度演化。同时，对四种经典差分格式（FTCS、FTFS、CTCS、Lax‑Wendroff）进行 von Neumann 稳定性分析，从理论上比较其适用范围。

---

## 2. Lax‑Wendroff 格式

### 2.1 数值格式推导

Lax‑Wendroff 方法是基于泰勒展开的二阶精度格式。对平流方程 $\rho_t = -C \rho_x$，有

$$
\rho(x, t+\Delta t) = \rho(x,t) + \Delta t   \rho_t + \frac{\Delta t^2}{2} \rho_{tt} + \mathcal{O}(\Delta t^3)
$$

利用原方程， $\rho_t = -C \rho_x$ ， $\rho_{tt} = C^2 \rho_{xx}$ ，因此

$$
\rho(x, t+\Delta t) = \rho - C \Delta t   \rho_x + \frac{C^2 \Delta t^2}{2} \rho_{xx} + \mathcal{O}(\Delta t^3)
$$

用中心差分近似空间导数：

$$
\rho_x \approx \frac{\rho_{j+1} - \rho_{j-1}}{2\Delta x}, \qquad \rho_{xx} \approx \frac{\rho_{j+1} - 2\rho_j + \rho_{j-1}}{\Delta x^2}
$$

代入并整理，得到离散格式（令 $r = C \Delta t / \Delta x$）：

$$
\rho_j^{n+1} = \rho_j^n - \frac{r}{2} (\rho_{j+1}^n - \rho_{j-1}^n) + \frac{r^2}{2} (\rho_{j+1}^n - 2\rho_j^n + \rho_{j-1}^n)
$$

亦可写为通量形式（如代码中所用）。该格式为二阶精度，且在 $|r| \le 1$ 时稳定。

### 2.2 代码实现

函数 `lax_wendroff(rho, C, dt, dx)` 实现单步更新。边界处采用零梯度条件（外推），即 $\rho_0 = \rho_1$ ， $\rho_{N-1} = \rho_{N-2}$。

---

## 3. 初始条件与模拟

定义三种初始密度分布：

- **情况 (a)**：仅在区间 $(L/4, L/2)$ 内有抛物线形分布

$$
\rho_0(x) = \frac{4}{L^2} (x - L/4)^2, \quad x \in (L/4, L/2)
$$

- **情况 (b)**：在 $(L/4, 3L/4)$ 内有三角形分布

$$
\rho_0(x) = \frac{4}{L} \left( \frac{L}{4} - \left| x - \frac{L}{2} \right| \right), \quad x \in (L/4, 3L/4)
$$

- **情况 (c)**：同 (b)，但波速取 $C = -1$，即波向左传播。

取网格间距 $\Delta x = 1$，时间步长 $\Delta t = 0.5$，Courant 数 $r = C \Delta t / \Delta x = 0.5$，满足稳定性条件。总步数 $n_t = 800$，模拟总时长 $400$ 个时间单位。

每 5 个时间步保存一帧，生成动画并保存为 MP4 文件（情况 (a) 保存，其余显示但不保存）。动画清晰展示密度波沿指定方向的传播和形状保持情况。

---

## 4. von Neumann 稳定性分析

对平流方程 $u_t = -c u_x$，考虑均匀网格，设解为单个傅里叶模态 $u_j^n = \xi^n e^{i k j \Delta x}$，其中 $\xi = e^{-i \omega \Delta t}$ 为放大因子。将数值格式代入，得到 $\xi$ 与 $k$ 的关系，并要求 $|\xi| \le 1$（von Neumann 条件）。

### 4.1 FTCS (Forward Time Centered Space)

格式：

$$
u_j^{n+1} = u_j^n - \frac{c \Delta t}{2 \Delta x} (u_{j+1}^n - u_{j-1}^n)
$$

代入模态：

$$
\xi = 1 - i \alpha \sin \theta, \quad \alpha = \frac{c \Delta t}{\Delta x}, \quad \theta = k \Delta x
$$

模方：

$$
|\xi|^2 = 1 + \alpha^2 \sin^2 \theta \ge 1
$$

故 $|\xi| > 1$（除非 $\alpha=0$） ，**无条件不稳定**。

### 4.2 FTFS (Forward Time Forward Space)

格式：

$$
u_j^{n+1} = u_j^n - \alpha (u_j^n - u_{j-1}^n)
$$

代入模态：

$$
\xi = 1 - \alpha (1 - e^{-i\theta}) = 1 - \alpha + \alpha e^{-i\theta}
$$

稳定性条件为 $0 \le \alpha \le 1$ 且要求 $|\xi| \le 1$ ，即 $0 \le \alpha \le 1$ 时稳定（但实际为条件稳定，且仅对右行波）。

### 4.3 CTCS (Centered Time Centered Space，即 Leap‑Frog)

格式：

$$
u_j^{n+1} = u_j^{n-1} - \alpha (u_{j+1}^n - u_{j-1}^n)
$$

代入模态，得 $\xi$ 满足：

$$
\xi^2 - 1 = -2 i \alpha \sin \theta   \xi
$$

解得：

$$
\xi = -i \alpha \sin \theta \pm \sqrt{1 - \alpha^2 \sin^2 \theta}
$$

当 $|\alpha \sin \theta| \le 1$ 时， $|\xi| = 1$ （中性稳定）；否则有一个根模大于 1。因此条件稳定： $|\alpha| \le 1$ 。

### 4.4 Lax‑Wendroff

格式（如前所述）：

$$
u_j^{n+1} = u_j^n - \frac{\alpha}{2} (u_{j+1}^n - u_{j-1}^n) + \frac{\alpha^2}{2} (u_{j+1}^n - 2u_j^n + u_{j-1}^n)
$$

代入模态，得：

$$
\xi = 1 - i \alpha \sin \theta - \alpha^2 (1 - \cos \theta)
$$

利用 $1 - \cos \theta = 2 \sin^2(\theta/2)$，可写为：

$$
\xi = 1 - 2\alpha^2 \sin^2(\theta/2) - i \alpha \sin \theta
$$

模方：

$$
|\xi|^2 = 1 - 4 \alpha^2 (1 - \alpha^2) \sin^4(\theta/2)
$$

当 $|\alpha| \le 1$ 时， $|\xi|^2 \le 1$ ，**稳定**（且为二阶精度）。当 $|\alpha| > 1$ 时不稳。

---

## 5. 结果与讨论

- 模拟结果显示，Lax‑Wendroff 格式能够无耗散地平移初始波形（无幅值衰减），仅存在轻微的相位误差（数值色散），波形基本保持。
- 情况 (a) 中抛物线向右平移，情况 (b) 三角形波右移，情况 (c) 左移，均符合预期。
- 稳定性分析表明，FTCS 完全不稳定，FTFS 条件稳定（受风向限制），CTCS 条件稳定且无耗散，Lax‑Wendroff 条件稳定且具有较好的精度。在实际计算中，Lax‑Wendroff 是平流问题的常用选择。

---

## 6. 代码结构与运行说明

代码包含以下主要部分：

- 参数设置： $L$ 、 $C$ 、 $\Delta x$ 、 $\Delta t$ 、总步数等。
- `lax_wendroff` 函数：执行单步 Lax‑Wendroff 更新。
- `rho0_a`、`rho0_b`：定义初始条件。
- `simulate`：循环迭代，存储历史密度。
- `create_animation`：生成动画并保存为 MP4 或嵌入 Jupyter 显示。
- 主程序依次模拟三种情况，情况 (a) 保存动画为 `traffic_case_a.mp4`，其余在 Notebook 中显示。

运行环境需安装 `numpy`、`matplotlib`、`imageio_ffmpeg` 和 `IPython`。在 Jupyter Notebook 中执行即可看到动画。若需保存所有动画，可取消相关注释。

---
