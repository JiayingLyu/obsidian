# 大模型 / 多模态大模型后训练实习八股讲义

> 面向 LLM、VLM、VLA 后训练、SFT、RLHF、DPO、GRPO、Alignment 算法实习。
>
> 本文按照 `llm_post_training_interview_checklist.md` 的准备顺序展开。目标不是罗列术语，而是建立一条统一主线：**模型先学习数据分布，再学习指令行为，再从偏好或环境奖励中优化序列级目标；数据、评测和训练系统决定这些算法能否真正工作。**

## 如何使用这份讲义

每个主题至少完成四种复述：

1. **一句话答案**：面试官刚问完时先给结论。
2. **原理与公式**：说明为什么，而不是只说“效果更好”。
3. **工程落地**：训练时具体有哪些张量、模型、指标和故障。
4. **项目联系**：把答案替换为自己项目中的真实配置、数据和结果。

文中代码强调面试可读性，省略了生产环境的分布式、异常处理和数值优化细节。

---

# 0. 先建立后训练的统一全景

给定上下文或 prompt `x`，自回归模型定义：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t})
$$

不同训练阶段本质上在回答不同问题：

| 阶段 | 数据 | 核心目标 | 模型学到什么 |
|---|---|---|---|
| 预训练 | 大规模原始文本/多模态数据 | 最大化下一个 token 似然 | 世界知识、语言与通用模式 |
| Continued Pretraining | 领域原始数据 | 同样是 next-token loss | 领域分布、术语和知识 |
| SFT | `指令 → 理想回答` | 模仿示范回答 | 指令遵循、格式、基本行为 |
| Reward Model | `chosen > rejected` | 学习偏好排序 | 一个可微的代理奖励 |
| DPO | 离线偏好对 | 直接提高 chosen 相对概率 | 离线偏好对齐 |
| PPO/GRPO | 在线 rollout + reward | 最大化期望序列奖励并限制策略漂移 | 探索并发现高奖励行为 |

把它们连起来看：

- **预训练**解决“模型有没有能力生成”。
- **SFT**解决“模型是否知道应当怎样回答”。
- **偏好优化/RL**解决“多个看起来合理的回答，哪个整体更好”。
- **数据和奖励**规定什么叫“好”。
- **评测**判断训练是否真的改进了目标，而不是钻奖励漏洞。
- **系统**决定 rollout、前反向和模型同步能否以可接受成本运行。

---

# 第一阶段：数学、深度学习与 PyTorch 基础

## 1.1 概率、期望与贝叶斯公式

### 概率、条件概率、联合概率

联合概率表示两个事件同时发生：`P(A,B)`；条件概率表示已知 `B` 后 `A` 的概率：

$$
P(A\mid B)=\frac{P(A,B)}{P(B)}
$$

由 `P(A,B)=P(A|B)P(B)=P(B|A)P(A)` 得贝叶斯公式：

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
$$

在机器学习中，`P(A)` 可看作先验，`P(B|A)` 是似然，`P(A|B)` 是后验。最大似然只优化似然；MAP 还乘上参数先验，因此可把某些正则化理解成先验约束。

### 期望、方差、协方差

$$
\mathbb E[X]=\sum_xp(x)x,
\qquad
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2]
$$

期望是长期平均，方差衡量随机变量波动。协方差

$$
\operatorname{Cov}(X,Y)=\mathbb E[(X-\mathbb EX)(Y-\mathbb EY)]
$$

衡量两个变量同向或反向变化，但受量纲影响；相关系数再除以标准差。RL 中我们不仅关心策略梯度是否无偏，也关心方差是否过大，因为高方差会导致训练不稳定。baseline、advantage normalization、GRPO 组内标准化都在处理方差。

## 1.2 最大似然、交叉熵和 next-token prediction

给定数据集 `D={(x_i,y_i)}`，MLE 选择使观察数据概率最大的参数：

$$
\theta^*=\arg\max_\theta\prod_i p_\theta(y_i\mid x_i)
=\arg\max_\theta\sum_i\log p_\theta(y_i\mid x_i)
$$

由于优化器默认最小化，取负得到 Negative Log-Likelihood。对 one-hot 标签，交叉熵为：

$$
H(q,p_\theta)=-\sum_kq_k\log p_{\theta,k}=-\log p_\theta(y)
$$

所以分类交叉熵就是 NLL；自回归语言模型把序列概率分解为各 token 条件概率，训练 loss 为：

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N}\sum_{i,t}m_{i,t}
\log p_\theta(y_{i,t}\mid x_i,y_{i,<t})
$$

其中 `m` 是 loss mask。**语言模型训练是 MLE**，因为它最大化训练语料每个真实 token 在其历史条件下的概率。

## 1.3 熵、交叉熵和 KL 散度

- 熵 `H(p)=-Σp log p`：分布自身的不确定性。
- 交叉熵 `H(p,q)=-Σp log q`：真实分布为 `p` 时，用 `q` 编码的平均代价。
- KL 散度：

$$
D_{KL}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}=H(p,q)-H(p)
$$

训练时真实数据分布 `p_data` 固定，因此最小化交叉熵等价于最小化 `KL(p_data || p_model)`。

### 为什么 KL 不对称

`KL(p||q)` 的期望在 `p` 下计算，而 `KL(q||p)` 在 `q` 下计算。若 `p` 在某处有质量但 `q=0`，前者趋于无穷；反过来惩罚区域不同。直觉上：

- `KL(p||q)` 更怕漏掉 `p` 的模态，倾向覆盖。
- `KL(q||p)` 更怕把质量放在 `p` 很小的位置，常更 mode-seeking。

这个直觉不是所有有限参数模型中的严格结论，但可帮助理解 reference KL 对策略分布的约束。

## 1.4 Jensen 不等式与 ELBO

对凹函数 `f`：

$$
f(\mathbb E[X])\geq \mathbb E[f(X)]
$$

`log` 是凹函数。含潜变量 `z` 时，直接算 `log p(x)=log ∫p(x,z)dz` 困难，引入近似后验 `q(z|x)`：

$$
\log p(x)
=\log\mathbb E_{q(z|x)}\left[\frac{p(x,z)}{q(z|x)}\right]
\geq
\mathbb E_q\left[\log\frac{p(x,z)}{q(z|x)}\right]
$$

右边是 ELBO。它等于 `log p(x) - KL(q(z|x)||p(z|x))`，所以最大化 ELBO 同时逼近证据和真实后验。后训练面试中通常只要求理解“通过 Jensen 得到可优化的下界”。

## 1.5 重要性采样

想在目标分布 `p` 下求期望，却只有提议分布 `q` 的样本：

$$
\mathbb E_{x\sim p}[f(x)]
=\mathbb E_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right]
$$

要求 `q(x)>0` 覆盖 `p` 有质量的区域。比值方差很大时估计不稳。PPO 的

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

就是重要性比值：样本来自旧策略，却评估新策略；clipping 防止少数极端比值主导更新。

## 1.6 计算图、链式法则与反向传播

若 `L=f(z), z=g(x)`：

$$
\frac{dL}{dx}=\frac{dL}{dz}\frac{dz}{dx}
$$

深度网络把运算记录成 DAG。forward 保存反向所需中间量；backward 从 loss 开始按逆拓扑序做 vector-Jacobian product，并把多条路径的梯度相加。反向传播不是数值求导，而是对计算图高效应用链式法则。

## 1.7 SGD、Momentum、Adam、AdamW

### SGD 与 Momentum

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

Momentum 对梯度做指数滑动平均：

$$
v_t=\mu v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t
$$

它能积累稳定方向、抑制来回振荡。

### Adam

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

经 bias correction 后：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

`m` 估计一阶矩，`v` 估计二阶原始矩，逐参数调整有效学习率。

### AdamW 为什么解耦 weight decay

若把 L2 正则梯度 `λθ` 直接加到 Adam 梯度中，它也会被 `1/sqrt(v)` 自适应缩放，不再等价于统一比例的参数衰减。AdamW 单独执行：

$$
\theta\leftarrow(1-\eta\lambda)\theta
-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
$$

因此衰减强度更可控。bias、LayerNorm/RMSNorm 参数通常不做 weight decay。

## 1.8 学习率、batch size 与梯度噪声

- **Warmup**：训练初始 moments 尚不稳定、网络激活未适应，逐步提高 LR 避免大更新破坏参数。
- **Constant**：稳定但后期难精细收敛。
- **Linear decay**：简单，随步数线性降到目标值。
- **Cosine decay**：前期下降慢、后期平滑接近最小 LR。
- **更大 batch**：梯度估计方差通常更低、吞吐更高，但更新次数减少、可能泛化变差，需要调整 LR。
- **更小 batch**：噪声更大，有时有正则化效果，但不稳定且硬件利用率低。

不要机械套“batch 翻倍 LR 翻倍”；线性缩放是经验起点，后训练小数据、LoRA、RL 中需重新验证。

## 1.9 梯度消失、爆炸和 clipping

多层链式乘积中 Jacobian 特征值长期小于 1 会消失，大于 1 会爆炸。Residual、合理初始化、Normalization、门控激活能改善梯度传播。

全局 norm clipping：

$$
g\leftarrow g\cdot\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right)
$$

它限制单步异常更新，但不能修复错误数据、过高 LR、数值溢出或错误 loss。若每一步都触发 clipping，应继续找根因。

### Loss 不下降、NaN、突然发散排查顺序

1. **数据**：空样本、全 `-100`、token 越界、错误模板、极端长度、NaN reward。
2. **目标**：shift 是否正确，mask 是否正确，符号是否写反，除零/`log(0)`。
3. **数值**：FP16 overflow、softmax 未减最大值、reward/advantage 尺度过大。
4. **优化**：LR 过高、无 warmup、梯度累积未缩放、clip 位置不对。
5. **模型状态**：该冻结的参数是否在训练，训练/推理模式是否正确。
6. **分布式**：不同 rank 数据或 step 不一致、梯度未同步。

## 1.10 欠拟合、过拟合、正则化和 early stopping

| 现象 | 训练集 | 验证集 | 常见处理 |
|---|---|---|---|
| 欠拟合 | loss 高 | loss 高 | 增大容量/rank、训练更久、调高合理 LR、改善输入 |
| 过拟合 | 持续变好 | 先好后坏 | 更多数据、减少 epoch、weight decay/dropout、早停 |
| 分布偏移 | 训练很好 | 特定切片很差 | 重做数据覆盖、分层评测、重采样 |

Early stopping 根据独立验证指标选 checkpoint，不应用最终测试集调参。

## 1.11 激活、Normalization、Residual 与门控 FFN

| 组件 | 核心答案 |
|---|---|
| ReLU | `max(0,x)`，便宜但负半轴梯度为零，可能出现 dead neuron。 |
| GELU | 近似 `xΦ(x)`，平滑地按输入大小门控，早期 Transformer 常用。 |
| SiLU/Swish | `x·sigmoid(x)`，平滑、非单调，SwiGLU 使用它作门。 |
| LayerNorm | 对单个 token 的 hidden 维做中心化和方差归一化，再乘 `γ` 加 `β`。 |
| RMSNorm | 只按均方根缩放，不减均值，计算更简单；通常保留可学习 scale。 |
| Residual | `x+F(x)` 提供恒等路径，使深层网络更易优化，也保留原表示。 |
| Dropout | 训练时随机置零并缩放期望；推理时关闭。小数据有正则作用，大模型常低 dropout。 |
| BatchNorm | 依赖 batch 统计；语言序列长度、mask、微批大小变化大，且推理统计不便，故不适合作 Transformer 默认归一化。 |

GLU 形式为 `A(x) ⊙ σ(B(x))`；GeGLU 换 GELU，SwiGLU 换 SiLU：

$$
\operatorname{SwiGLU}(x)=\operatorname{SiLU}(xW_g)\odot(xW_u)W_d
$$

门控使网络按输入动态选择通道；为了控制参数量，使用 SwiGLU 时中间维度常相应调整。

## 1.12 PyTorch 高频细节

### Shape 与内存

- `reshape`：尽量返回 view；不满足内存布局时会复制。
- `view`：要求底层 stride 兼容，常在 `permute` 后失败。
- `transpose`：交换两个维度；`permute` 可重排任意维。
- `contiguous()`：按当前逻辑顺序生成连续副本。
- 广播从末维对齐，维度必须相等或其中一个为 1；它通常不真正复制数据，但后续 op 可能物化结果。

### Autograd

- 叶子张量通常是用户创建且 `requires_grad=True` 的参数，梯度累积在 `.grad`。
- `grad_fn` 记录非叶子张量如何计算而来。
- `detach()` 返回共享存储但脱离计算图的张量；原地修改要谨慎。
- `torch.no_grad()` 在上下文中不记录图，适合评测；`inference_mode()` 更激进、开销更低。
- PyTorch 默认梯度累积，因此每次 update 前必须清梯度。

### `train()` 与 `eval()`

它们切换 Dropout、BatchNorm 等模块行为，不会自动关闭梯度。评测通常同时使用 `model.eval()` 和 `torch.no_grad()`。

### Dataset、DataLoader、collate

Dataset 定义单样本；DataLoader 负责采样、多进程加载和 batch；collate 把不同长度样本 padding、构造 mask 和 labels。LLM 中数据错误最常发生在 collator/chat template，而不是模型 forward。

### DDP 梯度何时同步

DDP 在 backward 中通过 autograd hook 把梯度按 bucket 触发 AllReduce，并尽量与剩余 backward 重叠；不是等 `optimizer.step()` 才同步。梯度累积前几个 micro-batch 可用 `no_sync()` 避免不必要通信，最后一个 micro-batch 再同步。

### Hook 检查激活和梯度

`register_forward_hook` 可记录输出均值、方差、最大值；参数或张量 hook 可看梯度 norm。要及时移除 hook，避免泄漏；分布式时需标明 rank。

## 1.13 手撕代码

### 数值稳定的 Cross Entropy

```python
import torch

def cross_entropy(logits, targets, ignore_index=-100):
    # logits: [N, V], targets: [N]
    mask = targets.ne(ignore_index)
    x = logits[mask]
    y = targets[mask]
    log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return -log_probs.gather(1, y[:, None]).mean()
```

### RMSNorm

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # 在 float32 中算统计量更稳，再转回原 dtype
        rms_inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms_inv).to(x.dtype) * self.weight
```

### 梯度累积与裁剪

```python
model.train()
optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss / grad_accum_steps
    loss.backward()

    if (step + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

注意：如果最后剩余 micro-batch 不足 `grad_accum_steps`，需要单独 step；分布式时全局 batch 为 `micro_batch × grad_accum × world_size`。

### Padding 与 Attention Mask

```python
def collate_fn(samples, pad_id=0, ignore_index=-100):
    max_len = max(len(s["input_ids"]) for s in samples)
    input_ids, attention_mask, labels = [], [], []
    for s in samples:
        ids = s["input_ids"]
        n_pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(s["labels"] + [ignore_index] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }
```

### 第一阶段面试验收答案

“语言模型训练为什么是最大似然？”——模型把序列概率分解为每个 token 的条件概率；对真实 token 做交叉熵就是负对数似然，最小化它等价于最大化训练序列在模型下的概率。

“如何定位 NaN？”——先锁定首次出现 NaN 的 step 和张量，再按数据/目标函数/数值精度/学习率/分布式顺序排查；检查输入、loss 各项、logits/grad norm，尝试 FP32 局部计算、降低 LR、加入 clipping，但不能用 clipping 掩盖错误。

---

# 第二阶段：Transformer 与现代 LLM 架构

## 2.1 Self-Attention：公式、shape 与直觉

设输入 `X∈R^{B×L×d_model}`，`h` 个头，每头维度 `d_h=d_model/h`：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

reshape 后 `Q,K,V ∈ R^{B×h×L×d_h}`。每个头：

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M\in R^{B\times h\times L\times L}
$$

$$
A=\operatorname{softmax}(S),\qquad O=AV
$$

各头拼接后经 `W_O` 投影回 `d_model`。

直觉：`Q` 表示当前位置想找什么，`K` 表示每个位置提供什么索引，点积得到相关性，`V` 是实际聚合的内容。

### 为什么除以 `sqrt(d_h)`

若 Q/K 各维近似零均值单位方差，点积 `q·k` 方差约为 `d_h`。维度越大，logits 绝对值越大，softmax 越接近 one-hot，梯度变小。除以 `sqrt(d_h)` 把方差稳定在常数量级。

### Causal mask 与 padding mask

- causal mask 屏蔽未来位置 `j>i`，保证自回归因果性。
- padding mask 屏蔽补齐位置，防止真实 token 关注 PAD。
- 实现时通常把被屏蔽 logits 加一个足够小的负数，而不是 softmax 后再置零；后者会破坏归一化。

### 时间与空间复杂度

投影约 `O(BLd_model²)`；注意力分数和加权约 `O(BL²d_model)`。标准实现还物化 `L×L` 分数/概率，激活内存呈二次增长。长序列既可能受算力限制，也可能受 HBM 读写和激活内存限制。

### 训练为什么并行，生成为什么逐 token

训练时完整目标序列已知，用 causal mask 一次计算所有位置；每个位置的 label 是下一个 token。推理时第 `t+1` 个 token 的输入取决于刚采样的第 `t` 个 token，时间维存在真实依赖，只能逐步 decode；batch 和层内矩阵运算仍可并行。

## 2.2 MHA、MQA、GQA

| 结构 | Q heads | KV heads | 特点 |
|---|---:|---:|---|
| MHA | `h` | `h` | 表达能力强，KV Cache 最大 |
| MQA | `h` | `1` | 所有 query 共享一组 K/V，cache 最小，可能损失质量 |
| GQA | `h` | `g, 1<g<h` | 每组 Q 共享 K/V，质量与效率折中 |

KV Cache 大小近似：

$$
2\times B\times L\times n_{layers}\times n_{kv\_heads}\times d_h\times bytes
$$

前面的 `2` 对应 K 和 V。GQA/MQA 主要减少 `n_kv_heads`，所以明显降低 decode 内存和带宽。

## 2.3 Decoder-only Transformer 完整数据流

1. token IDs 经 embedding 得到 `X`。
2. 每层先 Norm，再做 causal self-attention，加 residual。
3. 再 Norm，做 gated FFN，加 residual。
4. 最终 Norm 后乘 LM head 得 logits。
5. 训练时 logits 与右移一位的 labels 做 CE；推理时取最后位置分布采样。

### Pre-LN 与 Post-LN

- Post-LN：`LN(x + F(x))`，原始 Transformer 使用；深层时梯度穿过每层 LN，训练更难。
- Pre-LN：`x + F(LN(x))`，residual 主干提供更直接的梯度路径，现代 LLM 常用。

Pre-LN 更稳不代表总是最终上限更高；不同架构还可能使用 sandwich norm、DeepNorm 等。

### Attention、FFN、Residual、Norm 各做什么

- Attention：在 token 之间路由信息。
- FFN：对每个 token 独立做非线性通道变换和知识存储。
- Residual：保留输入并提供梯度高速通道。
- Norm：控制激活尺度，稳定深层优化。

FFN 中间维度更大，是先把表示投影到高维特征空间、做非线性组合，再压回 hidden size；大量参数也位于 FFN。SwiGLU 用门控提高选择性。

### 三种 Transformer 架构

| 架构 | 注意力 | 适用 |
|---|---|---|
| Encoder-only | 双向 self-attention | 理解、分类、表示学习 |
| Encoder–Decoder | encoder 双向；decoder 因果 + cross-attention | 翻译、条件生成、输入输出分离 |
| Decoder-only | 单一因果序列 | 统一 next-token 目标、生成、in-context learning |

Decoder-only 主流的原因包括：目标统一、训练数据易扩展、生成接口统一、scaling 简洁；不是因为它理论上在所有任务都优于 encoder–decoder。

## 2.4 位置编码与 RoPE

纯 attention 对输入排列具有等变性，需要位置编码。绝对位置编码把位置向量加到 token embedding；问题是相对关系需网络自己学习，超出训练长度时也可能失效。

### RoPE

RoPE 把 Q/K 的每对维度看作二维向量，位置 `m` 对其旋转 `mθ_i`：

$$
R_m=
\begin{bmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta & \cos m\theta
\end{bmatrix}
$$

关键性质：

$$
(R_mq)^T(R_nk)=q^TR_{n-m}k
$$

因此 Q/K 各自编码绝对位置，但点积只依赖相对位移 `n-m`。RoPE 不作用于 V，因为位置主要通过 attention score 决定路由。

### 长度外推和 scaling

超过训练长度后，模型遇到未见过的旋转相位和相对距离，注意力模式可能失效。RoPE scaling 的共同目标是重新映射位置或频率，使更长序列的相位变化落在模型可处理范围：

- Position interpolation：压缩位置索引，简单但局部分辨率下降。
- NTK-aware scaling：按频率维调整，尽量兼顾局部和远程关系。
- YaRN：结合频率分段、插值和温度/尺度修正。

ALiBi 不旋转 Q/K，而是在 attention logits 中加入与距离相关的线性负偏置，天然表达“越远越不利”，外推较直接。

## 2.5 KV Cache、Prefill 与 Decode

自回归生成到第 `t` 步时，旧 token 的 K/V 不变。缓存每层旧 K/V，新一步只算当前 token 的 Q/K/V，再让当前 Q 关注缓存 K/V，把每步重复计算从“重跑全部前缀”降为“只处理新 token”。

- **Prefill**：一次处理完整 prompt；矩阵大、并行度高，通常 compute-bound。
- **Decode**：每次每请求处理一个 token；频繁读权重和 KV Cache，矩阵小，常 memory-bandwidth/调度受限。

KV Cache 随 `batch × sequence length × layers × kv heads × head dim` 线性增长，是并发和长上下文服务的重要瓶颈。

## 2.6 生成策略

- Greedy：每步取 argmax，确定性、便宜，但可能局部最优和重复。
- Beam search：保留累计 log-prob 最好的多条路径，适合翻译等目标较确定任务；开放生成常产生相似、保守文本。
- Temperature：`softmax(logits/T)`；`T<1` 更尖锐，`T>1` 更随机。
- Top-k：只在概率最高 k 个 token 中采样。
- Top-p：取累计概率达到 p 的最小候选集合，集合大小随分布变化。
- Repetition penalty：降低已出现 token 的相对分数；过强会破坏必要复用。
- Length penalty：修正 beam search 对短序列的偏好；不同实现公式需确认。

RL rollout 中温度不是单纯推理参数：过低导致组内样本相同、无学习信号；过高导致低质量轨迹和高方差。

## 2.7 MoE

Dense 模型每个 token 激活所有 FFN；MoE 放置多个 expert FFN，由 router 为每个 token 选择 top-k：

$$
y=\sum_{e\in TopK(x)}p_e(x)E_e(x)
$$

它可以增大总参数而不等比例增加每 token FLOPs，但引入路由、跨设备 All-to-All、负载不均和 expert 容量问题。

- router 输出 expert logits/probability。
- top-k expert 处理 token；shared expert 可吸收通用知识。
- load balancing loss 鼓励 token 和路由概率不要集中到少数 expert。
- 总参数量用于描述存储/容量；激活参数量更接近单 token 计算成本。
- expert parallelism 把 experts 分布在设备上，token 按路由 All-to-All 发往目标设备，再返回原布局。

## 2.8 Attention 与采样手撕代码

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # [B,H,L,Dh]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        if padding_mask is not None:  # [B,L], 1 表示有效
            scores = scores.masked_fill(~padding_mask[:, None, None, :].bool(), float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(y)
```

简化 top-k/top-p sampling：

```python
def sample_top_k_top_p(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        threshold = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), num_samples=1)
```

### 第二阶段面试验收答案

从 token ID 到下一 token：token embedding → 多层 Pre-Norm causal attention 与 SwiGLU FFN → final norm → LM head logits → temperature/top-k/top-p 处理 → softmax 概率 → 采样 token；训练时所有位置并行，推理时用 KV Cache 逐 token decode。

GQA 与 KV Cache 的关系：GQA 保持多个 query heads，但让一组 query 共享 K/V heads；cache 只需保存更少的 K/V，在通常较少损失质量的情况下提高 decode 并发和带宽效率。

---

# 第三阶段：Tokenizer、预训练与数据基础

## 3.1 字符、词和 Subword Tokenizer

| 单位 | 优点 | 缺点 |
|---|---|---|
| 字符/字节 | 几乎无 OOV、词表小 | 序列长，语义组合负担大 |
| 词 | 序列短、语义完整 | 词表巨大，未登录词和形态变化严重 |
| subword | 在词表和长度间折中，可组合生词 | 切分质量受语料与语言影响 |

现代 LLM 常用 subword 或 byte-aware tokenizer，使任意字符串都可编码。

## 3.2 BPE 训练与编码

1. 从字符/字节级符号开始统计词频。
2. 反复寻找频率最高的相邻符号对并合并。
3. 将每次 merge 记录为有顺序的规则，直到达到词表大小。
4. 编码新文本时按所学规则合并。

高频片段成为单 token，低频词拆成更小单元。BPE 的目标主要是压缩与覆盖，不保证每个 token 是语言学词素。

WordPiece 通常按对语言模型似然/关联度的贡献选 merge；Unigram 从大词表出发，给子词概率并迭代删减，用动态规划找最优切分。面试重点是三者方向：BPE 自底向上合并，Unigram 自顶向下裁剪。

## 3.3 Special Tokens 与 Chat Template

- BOS：序列开始；是否需要由模型训练约定决定。
- EOS：序列或 assistant 回答结束；训练遗漏会导致不停生成。
- PAD：batch 对齐；不应计入 loss，也不应被 attention 使用。
- UNK：无法编码的符号；byte fallback 模型可极少使用。
- role/control tokens：区分 system/user/assistant、图像等边界。

Chat template 是训练时把结构化消息序列化为 token 的协议。模板、role token、EOS 与模型预训练/SFT 不一致，会造成分布偏移、角色混淆、模型复述 user、无法停机。**模板不是展示格式，而是模型输入分布的一部分。**

## 3.4 词表大小与中英文压缩率

大词表使序列更短，但 embedding/LM head 参数和 softmax 计算变大，低频 token 训练不足；小词表参数少但序列更长，attention 和 decode 成本增加。

如果 tokenizer 主要在英文语料训练，英文常见词片段可被压成较少 token，而中文字符、词组或稀有汉字可能拆得更多；具体压缩率必须实测，不能认为“一个汉字永远一个 token”。

扩词表时需扩展 input embedding 和 LM head（若权重绑定则保持绑定），新行可随机初始化、用相关 token 均值初始化，再通过 continued pretraining/SFT 学习；只改 tokenizer 不训练新 embedding 会使新 token 无意义。

## 3.5 自回归预训练与 Perplexity

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

token-level loss 是目标 token 负 log-prob 的平均。Perplexity：

$$
PPL=\exp\left(-\frac1N\sum_t\log p(x_t\mid x_{<t})\right)=e^{CE}
$$

可理解为模型平均面临的有效分支数，越低通常越好。但：

- 不同 tokenizer 的 PPL 不可直接比较。
- PPL 衡量文本拟合，不直接衡量指令遵循、事实性、安全性和推理正确率。
- SFT 只对 assistant tokens 算 loss 时，与全语料 PPL 含义不同。

## 3.6 Pretraining、Continued Pretraining、SFT

- Pretraining：从广泛原始序列学 `p(text)`，获得通用表征和生成能力。
- Continued Pretraining：仍用无监督 next-token loss，但数据转向领域，适合注入术语、语言风格和知识分布。
- SFT：以任务输入和理想输出为示范，通常只优化回答 token，主要学习条件行为。

“预训练学知识，SFT 学行为”是有用近似，不是硬边界：SFT 也会写入事实，continued pretraining 也会改变行为；区别在数据结构与监督信号。

## 3.7 Scaling law 与算力权衡

在一定区间内，模型 loss 随参数、数据、算力增加呈可预测的幂律下降；给定算力存在模型大小和训练 token 数的较优配比。结论不是“模型越大永远越好”，而是算力固定时，过大模型但数据不足会 undertrain，过小模型则容量不足。后训练也有自己的数据/采样/更新预算：增加 rollout 数可能比增加 update epoch 更有价值。

## 3.8 预训练数据治理

标准 pipeline：解析 → 语言/模态识别 → 规则清洗 → 质量分类 → 去重 → 风险与隐私过滤 → 数据配比 → tokenize/packing → 版本冻结。

- 文档级 exact hash：去完全重复。
- 段落级去重：避免模板、转载片段重复。
- n-gram + MinHash/LSH：近似 Jaccard，相似文档可扩展去重。
- contamination：训练数据含 benchmark 题目、答案或近似版本，导致虚高。
- 配比：按领域/语言设置采样权重，而不是简单按原始数量混合。
- curriculum：先易后难或动态提高困难样本，但必须验证是否真正改善而非延迟困难训练。

数据质量和规模不是二选一：低质量数据会消耗算力、复制偏差；过度过滤又会损失覆盖和多样性。合理答案应说明质量阈值、保留率、切片评测与消融。

---

# 第四阶段：SFT 与参数高效微调

## 4.1 SFT 样本如何组织

典型消息：

```text
<system> 你是……
<user> 问题/任务输入
<assistant> 理想回答 <eos>
```

instruction 描述任务；input 提供实例条件；response 是监督输出；system 定义长期行为/角色。它们最终经 chat template 变成一个 token 序列。

### Labels、右移与 `-100`

多数 causal LM 内部会把 logits `[:, :-1]` 与 labels `[:, 1:]` 对齐。labels 中 prompt、padding 以及不希望监督的位置设 `-100`，PyTorch CE 忽略它们。

只对 assistant tokens 计算 loss 的理由：目标是学习回答条件分布，不需要让模型背诵用户 prompt；还可避免不同 prompt 长度主导 loss。若全序列 loss，模型也学习 system/user token 的复现，可能浪费容量，但在 continued pretraining 风格任务中可合理。

多轮对话一般对每轮 assistant 内容监督，user/system mask；是否监督历史 assistant 取决于训练目标。关键是每个 assistant 起止 token、EOS 均正确包含。

## 4.2 Padding、截断与 Packing

- 训练常右 padding；批量生成 decoder-only 常左 padding，使各序列最后一个有效位置对齐。具体实现看模型 position IDs 和 kernel。
- truncation 不应盲目截尾：可能只留下问题没有答案，或删掉 EOS。应先看长度分布，并按任务保存关键信息。
- max length 取覆盖率、显存和任务需求折中，报告被截断比例。
- packing 把多条短样本拼入固定长度，提高 token 利用率。必须用 segment-aware mask 或正确边界/EOS，避免样本间错误 attention；loss 也必须按各样本 assistant 区间 mask。

长样本若按 token 平均，会贡献更多梯度；若先对每个样本平均再对 batch 平均，则样本权重相等。面试应明确使用哪一种。

## 4.3 SFT loss 下降但生成变差

可能原因：

1. 训练 loss 只衡量 teacher forcing 下的 next-token，不等价于自由生成质量。
2. 数据模板、EOS 或 mask 错误，模型学到错误格式。
3. 小数据过拟合/灾难性遗忘。
4. 训练与评测 decoding 参数不同。
5. 长度、领域分布与验证集不一致。
6. 只学高概率措辞，输出更模板化但任务正确率下降。
7. checkpoint 选择只看 loss，没有看任务指标。

## 4.4 数据质量、数量、多样性和配比

- 质量：答案正确、遵循指令、格式一致、无泄漏。
- 数量：提高覆盖，但重复数据的边际收益低。
- 多样性：覆盖任务、语言、难度、风格和长度。
- 配比：防止大数据源吞没小但重要的数据源。

灾难性遗忘的处理：混入通用 replay 数据、降低 LR/epoch、使用 LoRA 或冻结部分层、加入 reference KL/蒸馏约束、同时评测通用回归集。LoRA 只限制更新子空间，不保证不会遗忘。

## 4.5 Full Fine-tuning、LoRA、QLoRA

| 方法 | 基座权重 | 可训练参数 | 主要优点 | 主要风险 |
|---|---|---|---|---|
| Full FT | BF16/FP32，更新 | 全部 | 容量最大 | 显存和存储高、易遗忘 |
| LoRA | 冻结 | 低秩矩阵 | 显存低、模块化、可合并 | rank/插入层限制容量 |
| QLoRA | 4-bit 冻结基座 | BF16/FP16 LoRA | 基座显存更低 | 量化噪声、kernel/吞吐限制 |

LoRA 假设任务适配所需的权重增量具有低内在秩：

$$
W'=W+\Delta W,
\qquad
\Delta W=\frac{\alpha}{r}BA
$$

若 `W∈R^{d_out×d_in}`，可令 `A∈R^{r×d_in}`、`B∈R^{d_out×r}`。常把 B 零初始化，使训练开始时 `ΔW=0`，保持基座行为。

- rank `r`：容量与参数/显存的核心旋钮。
- alpha：更新尺度，常通过 `alpha/r` 缩放。
- dropout：仅在 LoRA 分支输入上正则化。
- target modules：常见 `q/k/v/o_proj` 与 `gate/up/down_proj`；只插 Q/V 更省，但复杂任务可能容量不足。
- merge：推理前把 `ΔW` 加入 W，可无额外 LoRA matmul；但量化基座合并需注意重新量化精度。

### QLoRA

基座权重以 4-bit 存储并冻结；forward 时按块反量化到计算 dtype，梯度穿过反量化运算到 LoRA，但不更新 4-bit 基座。LoRA 权重和优化器状态仍使用较高精度。

- NF4：针对近似正态分布权重设计非均匀 4-bit codebook。
- Double quantization：进一步量化每块的量化 scale 常数，减少元数据。
- Paged optimizer：借助统一内存缓解长序列/检查点造成的瞬时显存峰值；它不是 PagedAttention。

rank 太小会欠拟合，训练 loss 降不下或复杂能力不足；rank 太大增加显存、通信和过拟合风险，并不保证收益。应做 rank×target-module 消融。

Prefix/Prompt Tuning 在输入/各层加入可学习虚拟 token；Adapter 在层内插小瓶颈网络。它们通常推理时有额外路径，而 LoRA 可 merge。

## 4.6 SFT 调参与排查

### 超参数逻辑

- LR：Full FT 通常小于 LoRA；数据少/模型大时更保守。
- Epoch：以验证指标和数据重复率决定，不按固定 3 epoch 迷信。
- Effective batch：影响噪声和更新次数；报告 micro batch、accumulation、world size。
- Warmup：常按总 steps 比例或固定步数，训练很短时比例尤其重要。
- Weight decay：主要作用于可训练矩阵；Norm/bias 通常排除。

### 高频故障

- 格式崩坏：检查模板、role token、数据混合、EOS、解码 stop 条件。
- 重复生成：数据重复/模板化、温度过低、EOS 弱、过拟合、repetition penalty 仅是表面缓解。
- EOS 不学习：EOS 未加入 labels、被 truncation、mask 成 `-100`、训练/推理 EOS ID 不一致。
- 回答截断：max_new_tokens、训练数据尾部截断、stop string 误触发。
- 通用能力下降：混入 replay、减小更新、分领域 adapter、加 reference 约束。
- 多数据源不平衡：按数据源记录 loss/梯度/采样次数；做温度采样、上限采样或 loss reweight。

SFT baseline 至少包含：基座/原 instruct 模型、同数据同评测的 LoRA SFT、关键数据源/模板/rank/epoch 消融，并报告通用与领域两类指标。

## 4.7 SFT 手撕代码

### Assistant-only mask

最稳妥做法是在模板化时记录每段 token span，而不是对 token IDs 搜索一段文本，因为 BPE 边界可能改变。

```python
def build_labels(input_ids, assistant_spans, ignore_index=-100):
    labels = [ignore_index] * len(input_ids)
    for start, end in assistant_spans:  # [start, end)
        labels[start:end] = input_ids[start:end]
    return labels
```

### LoRA Linear

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.dropout(x))) * self.scale
```

### Causal SFT loss

```python
def causal_lm_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

### SFT pipeline 标准答案

原始数据审计与切分 → 固定 chat template → tokenize 并生成 assistant-only labels → 统计长度/截断率 → 构建基座评测 → 选择 Full/LoRA/QLoRA → 记录训练/验证 loss、grad norm 和吞吐 → 按领域与通用指标选 checkpoint → 消融和失败分析 → 固化数据、配置、代码和模型版本。

“为什么用 SFT 而不是 continued pretraining？”——如果目标是让模型对明确指令产生结构化答案，SFT 的监督直接作用于目标行为；continued pretraining 更适合吸收大量无标注领域文本。若既缺领域知识又缺行为，通常先 continued pretraining，再 SFT。

---

# 第五阶段：强化学习基础

## 5.1 MDP、Trajectory 与 Return

马尔可夫决策过程可写作 `(S,A,P,R,γ)`：

- state `s_t`：决策所需状态。
- action `a_t`：智能体采取的动作。
- transition `P(s_{t+1}|s_t,a_t)`：环境转移。
- reward `r_t`：当前反馈。
- trajectory `τ=(s_0,a_0,r_0,...,s_T)`：完整交互轨迹。
- return：从 t 开始的折扣累计奖励：

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

`γ<1` 让远期奖励权重下降并在无限时域保证和收敛；有限长度 LLM 任务常取接近 1，若只有结尾奖励，所有 response token 可能共享该序列回报。

区分三个概念：step reward 是单步反馈；episode reward 常指整条轨迹最终分数；return 是从某时刻累积的未来 reward。只有终局 reward 时三者不要混用。

## 5.2 Policy、V、Q 与 Advantage

$$
V^\pi(s)=\mathbb E_\pi[G_t|s_t=s]
$$

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t|s_t=s,a_t=a]
$$

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

`V` 是到达状态后的平均前景，`Q` 是特定动作的前景，`A` 衡量该动作相对该状态平均水平好多少。策略梯度乘 advantage 的直觉是：提高“比预期好”的动作概率，降低“比预期差”的动作概率。

Bellman expectation equation：

$$
V^\pi(s)=\mathbb E_{a\sim\pi,s'\sim P}[r(s,a)+\gamma V^\pi(s')]
$$

它把长期价值写成一步 reward 加下一状态价值，是 TD、critic 的基础。

## 5.3 On-policy、Off-policy、Monte Carlo、TD

| 维度 | 方法 | 特点 |
|---|---|---|
| 数据策略 | On-policy | 用当前/非常接近当前策略的数据更新，分布匹配但 rollout 贵 |
| 数据策略 | Off-policy | 可复用旧策略/其他策略数据，效率高但要处理分布偏移 |
| 价值目标 | Monte Carlo | 等整条轨迹后用真实 return；低偏差、高方差 |
| 价值目标 | TD | 用 `r+γV(s')` bootstrap；方差低但引入 critic 偏差 |

LLM 的 DPO 是离线偏好优化；PPO/GRPO 通常希望近似 on-policy。异步 rollout 会自然带来一定 off-policy/staleness。

## 5.4 Exploration、Entropy 与 Bias–Variance

策略熵：

$$
H(\pi(\cdot|s))=-\sum_a\pi(a|s)\log\pi(a|s)
$$

熵高表示分布较平、有探索；熵低表示策略确定。entropy bonus 可防止过早塌缩，但过大使策略长期随机。LLM 中采样温度影响实际 rollout 探索，训练 objective 中 entropy 系数则影响参数更新，两者不是同一旋钮。

MC return 接近真实结果但样本波动大；critic/TD 降方差但若价值估计错误会带偏。GAE 的 `λ`、GRPO 的组内 baseline、RLOO 的 leave-one-out baseline 都是在偏差—方差—成本之间折中。

## 5.5 Policy Gradient 推导

目标：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
=\sum_\tau p_\theta(\tau)R(\tau)
$$

利用 `∇p=p∇log p`：

$$
\nabla_\theta J
=\mathbb E_{\tau}\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right]
$$

环境转移不依赖策略参数，所以：

$$
\log p_\theta(\tau)=C+\sum_t\log\pi_\theta(a_t|s_t)
$$

于是：

$$
\nabla_\theta J
=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)G_t\right]
$$

这就是 log-derivative trick 和 REINFORCE。

### Baseline 为什么不引入偏差

只要 baseline `b(s)` 不依赖当前采样 action：

$$
\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a|s)]
=b(s)\nabla\sum_a\pi(a|s)=b(s)\nabla1=0
$$

所以用 `G-b(s)` 不改变期望梯度，但可减少所有动作共享的状态难度波动。最佳 baseline 与条件期望 return 相关，critic 近似它。

### Credit assignment

序列终局 reward 同时乘所有 token log-prob，只知道“整段好/坏”，不知道具体哪一步负责。过程奖励、critic、GAE、可验证中间步骤可以提供更细信号，但过程标注可能有偏差，被优化后也可能出现新的 hacking。

## 5.6 Actor–Critic 与 GAE

Actor 是 policy，critic 学 `V(s)`。TD residual：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

GAE：

$$
\hat A_t^{GAE(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
$$

- `λ=0` 接近一步 TD，方差低、偏差更依赖 critic。
- `λ→1` 接近 MC advantage，偏差低、方差高。
- critic 用 return/value target 训练，actor 用 advantage 更新。

## 5.7 RL 概念映射到 LLM

给定 prompt `x`：

- state：`(x,y_{<t})`。
- action：下一个 token `y_t`。
- policy：语言模型条件分布 `πθ(y_t|x,y_<t)`。
- trajectory：完整 response token 序列。
- environment：任务执行器、工具、用户/模拟器或只在终局打分的静态环境。
- reward：RM、规则、单元测试、数学 verifier、仿真反馈等。

sequence reward `R(x,y)` 可直接作为每个 response token 的 return，得到：

$$
\mathcal L_{PG}=-\frac1T\sum_t A_t\log\pi_\theta(y_t|x,y_{<t})
$$

语言词表使 action space 达数万；每条轨迹的组合空间指数增长。模型已有 SFT/pretrained prior 非常重要，否则从零探索几乎不可能。temperature 越高，探索越广但平均轨迹质量和 reward SNR 可能下降。

token-level KL 常对采样 token 使用 `logπθ(y_t)-logπref(y_t)` 的样本估计；沿序列求和/平均形成 sequence KL 估计。精确 KL 需要对完整词表求和，成本更高。

## 5.8 RL 手撕代码

```python
def discounted_returns(rewards, gamma=1.0):
    out = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out

def reinforce_loss(response_logps, returns, mask):
    # logps/returns/mask: [B,T]
    return -((response_logps * returns.detach()) * mask).sum() / mask.sum()

def normalize_advantage(adv, mask, eps=1e-8):
    valid = adv[mask.bool()]
    mean, std = valid.mean(), valid.std(unbiased=False)
    return (adv - mean) / (std + eps)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # values 长度 T+1
    T = rewards.size(0)
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        adv[t] = last
    return adv, adv + values[:-1]
```

“为什么 SFT 不能直接优化正确率？”——SFT 对给定示范逐 token 做可微 MLE；自由生成后的正确率、执行成功和闭环指标是离散的序列函数，不能直接沿采样结果反传。Policy Gradient 用 `∇logπ · reward` 给出期望奖励的无偏梯度估计。

---

# 第六阶段：偏好数据与 Reward Model

## 6.1 偏好数据的四种来源

对同一 prompt `x`，`y_w` 是 chosen，`y_l` 是 rejected。候选回答最好来自目标模型或相近策略，否则容易出现训练—部署分布差异。

- 人工标注：最贴近真实偏好，但昂贵且有主观噪声。
- 规则：客观、便宜、可复现，但只覆盖可形式化属性。
- 强模型/LLM judge：可规模化，继承 judge 偏差和自偏好。
- rejection sampling：同一 prompt 采样多个，按 verifier/RM 选最好与较差样本；会偏向采样分布中已有能力。

### Pointwise、Pairwise、Listwise

- pointwise：每个回答独立打分，标注简单但标尺漂移。
- pairwise：只判断 A/B 哪个好，人更容易一致；DPO/RM 常用。
- listwise：同时排序多项，信息更丰富但认知负担和一致性问题更大。

### Pair 难度

- 差距过大：容易学，梯度很快饱和，只学到粗糙规则。
- 差距过小：有细粒度信息，但标签噪声和争议大。
- hard negative：表面好、关键处错，能训练细辨别能力；若人工也难判断，会放大噪声。

合理数据集混合易、中、难 pair，并按领域/长度/错误类型切片评测。

## 6.2 偏好数据的常见偏差

- Position bias：标注者/judge 偏爱先展示的答案。随机交换顺序并检查一致性。
- Length bias：更长答案看似详细，或短答案看似简洁。按长度匹配/分桶评测。
- Style bias：偏爱标题、引用、礼貌语，而非事实正确性。rubric 分维度评分。
- Self-preference：judge 偏好与自己风格或来源相似的答案。
- 标注噪声：多人意见不同、标准模糊。记录 annotator，算一致性并审计低 margin 样本。

Offline preference 固定、便宜，但模型学到新分布后数据不更新；online preference 对当前 rollout 标注，分布更匹配但成本高。迭代式 alignment 常交替 rollout、标注/打分、更新。

## 6.3 Reward Model 与 Bradley–Terry

通常在语言模型最后 hidden state/EOS 上加标量 head 得 `rφ(x,y)`。假设 chosen 胜出的概率：

$$
P(y_w\succ y_l|x)=\sigma(r_w-r_l)
$$

负对数似然：

$$
\mathcal L_{RM}=-\log\sigma(r_w-r_l)
$$

只约束差值，因此所有 reward 同加常数不改变概率，绝对零点不可识别；这也是训练中 reward 均值可能漂移的原因。可用中心化/正则化改善尺度，但排序才是主要目标。

### Sequence、Token、Outcome、Process Reward

- Sequence-level：整段一个标量，适合总体偏好但 credit assignment 粗。
- Token/step-level：每步或推理步骤给反馈，训练信号细但标注/对齐困难。
- Outcome RM：只评最终结果；客观 verifier 下更可靠，但错误过程偶然答对也得高分。
- Process RM：评每一步是否合理；可改善搜索和 credit，但会奖励“看起来像正确过程”的模式。
- Verifiable reward：单元测试、答案解析、规则检查、仿真指标等；低主观性、可在线大量使用，覆盖范围有限且解析器可被攻击。

## 6.4 Reward Hacking 与 Overoptimization

Reward 是真实目标的代理。策略在分布外寻找高分区域时，可能利用 RM/规则漏洞：输出更长、堆叠关键词、伪造格式、触发解析器、产生 judge 偏爱风格。随着代理 reward 继续上升，真实人工质量可能先升后降，这就是 overoptimization/Goodhart 效应。

缓解手段：

1. 多维 reward 与硬约束分开记录，不盲目线性加总。
2. 独立 held-out 人工/规则评测，不只看训练 reward。
3. KL/reference 约束和较小更新。
4. reward ensemble、不确定性惩罚或对抗样本。
5. 定期收集当前 policy 的新数据再训练 RM。
6. 审计 reward 高但任务失败的 top cases。

## 6.5 为什么 RM accuracy 高不等于 RL 有效

- 测试 pair 与 policy 优化后的分布不同。
- accuracy 不衡量 margin、校准和极端高 reward 区域。
- easy pairs 主导整体准确率，却不提供 near-policy 细粒度信号。
- RM 可学长度/格式 shortcut。
- RL 会主动寻找 RM 弱点，普通测试是被动 i.i.d. 评测。

因此还需按长度、领域、难度、生成模型来源切片；看 reward margin 分布、与人工分数相关性、policy best-of-N 与在线 RL 后真实质量。

## 6.6 RM Ensemble、Uncertainty 与 Calibration

多个不同 seed/架构/数据子集 RM 的均值可降低偶然噪声，方差可作为 epistemic uncertainty；策略在高分但模型分歧大的区域应谨慎。缺点是成本高、模型可能共享同一偏差。

Calibration 问“预测 0.8 胜率的 pair 是否约 80% chosen 获胜”。可做 reliability diagram、ECE、temperature scaling。排序训练不天然保证绝对 reward 可校准。

## 6.7 RM 评测清单

- overall/prompt-group accuracy。
- chosen-rejected reward margin 的均值、分位数。
- 长度差、领域、难度、来源模型切片。
- 交换 position 后的一致率。
- train/validation 与当前 policy rollout 的分布差异。
- 高 reward 失败样本、低 reward 正确样本的人工 taxonomy。
- OOD、对抗格式、重复文本、极端长度鲁棒性。

一套偏好数据构建标准答案：定义 rubric → 对每个 prompt 从目标附近的多种策略采样 → 规则先过滤明显坏样本 → 随机化顺序做人/模型 pairwise 标注 → 记录置信度和争议 → 按 prompt 隔离切分 → 做长度/领域/来源平衡 → 训练 RM/DPO → 用当前策略新 rollout 做分布外复测。

---

# 第七阶段：PPO 与经典 RLHF

## 7.1 SFT → RM → PPO 全流程

1. 用高质量 demonstrations 做 SFT，得到初始化策略 `π_SFT`。
2. 对 prompt 采样多个回答，收集 pairwise preference，训练 RM `rφ(x,y)`。
3. 初始化 trainable policy `πθ`、frozen reference `π_ref`，通常从 SFT checkpoint 开始；初始化 value model `Vψ`。
4. 用当前/旧 policy rollout。
5. 用 RM 给终局奖励，并加入相对 reference 的 KL penalty。
6. critic 估值、计算 GAE advantage。
7. 多个 minibatch epoch 优化 clipped policy loss、value loss、entropy/KL 项。
8. 更新 old policy/rollout 权重，重复。

## 7.2 五个“模型/策略”不要混淆

| 名称 | 是否训练 | 作用 |
|---|---|---|
| Policy `πθ` | 是 | 当前要优化的生成模型 |
| Old policy `π_old` | rollout 批次内冻结 | 产生样本并构成 PPO ratio 分母 |
| Reference `π_ref` | 通常全程冻结 | 锚定 SFT 行为，计算 KL/相对 log-prob |
| Reward model `rφ` | PPO 阶段通常冻结 | 对完整回答打代理分数 |
| Value model `Vψ` | 是 | 预测状态 return，构造 advantage |

old policy 约束“这一个 PPO batch 内更新别离采样分布太远”；reference 约束“整个 RL 过程中别离初始对齐模型太远”。它们概念上不同，即使训练开始时权重相同。

## 7.3 PPO Clipped Objective

样本来自 `π_old`，对每个 token：

$$
r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))
$$

$$
L^{clip}=\mathbb E_t\left[
\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)
\right]
$$

训练代码最小化其负数。

### `A>0` 与 `A<0`

- `A>0`：希望增大动作概率，即 `r>1`；但 `r>1+ε` 后收益被截住。
- `A<0`：希望减小动作概率，即 `r<1`；但 `r<1-ε` 后改善被截住。

`min` 配合 advantage 符号形成 pessimistic lower bound，防止样本上看起来特别有利的大更新。Clipping 不是严格 KL trust region；实际仍应监控 KL、clip fraction 和 ratio。

### Value loss 与 Entropy

$$
L_V=\frac12(V_\psi(s_t)-\hat R_t)^2
$$

value 也可做 clipping，避免 critic 一步大变。总 loss 常写：

$$
L=-L^{clip}+c_vL_V-c_eH(\pi)+\text{other regularization}
$$

Entropy 奖励探索；LLM 实践中也可能主要依赖采样温度而不显式加 entropy bonus。

## 7.4 PPO 中的 Reward 与 KL

常见 token reward：中间 token 只有 KL penalty，最后 token 再加 RM score：

$$
r_t^{total}=
-\beta(\log\pi_\theta(y_t|s_t)-\log\pi_{ref}(y_t|s_t))
+\mathbb 1[t=T]r_{RM}(x,y)
$$

三种相关做法：

- KL reward shaping：把样本 KL 作为逐 token 负奖励。
- KL loss/penalty：objective 中另加 KL 项。
- adaptive coefficient：实际 KL 高于 target 就增大 β，低于 target 就减小 β。

KL 太大：策略漂移、hacking/语言质量风险、LR/epoch/advantage 过大；可增 β、降 LR、减少 update epoch/clip。KL 太小：更新过弱、reward 信号小、β 过大、数据无区分或 clipping 太多。

Forward KL `KL(ref||policy)` 强烈惩罚 policy 在 ref 有质量处变零，偏覆盖；reverse KL `KL(policy||ref)` 惩罚 policy 走到 ref 低概率区域，更贴近 RLHF 的“不要离参考太远”。具体实现常是 sampled reverse-KL 估计。

## 7.5 为什么 PPO RLHF 显存和系统复杂

- 同时涉及 policy、reference、RM、value，多份模型权重。
- rollout 使用推理布局，训练使用训练布局，需要权重同步/重分片。
- policy/value 需要 activation、gradient、optimizer states。
- 生成长度不一，动态 batching 和负载平衡困难。
- on-policy 数据很快过期，不能无限复用。
- reward、GAE、mask、old log-prob、ref log-prob 都要与 token 精确对齐。

可让 value head 与 policy 共享 backbone 降权重成本，但会产生多任务干扰；也可完全独立，成本更高。

## 7.6 PPO 训练诊断

### Reward 上升但真实能力下降

首查 reward hacking/RM OOD、长度和格式 shortcut。看独立 verifier/人工胜率、reward 各分量、长度切片；增强 RM 数据、加 KL、早停或限制可被利用的格式。

### KL 快速升高

LR 或 advantage scale 过大、PPO epoch 太多、β 太小、old policy 过旧。检查 ratio/clip fraction、adv std、权重同步版本。

### Entropy 快速下降

策略过早确定，可能 reward 过尖、温度低、更新过强。提高采样多样性、减小更新、加 entropy/数据难度；但先确认任务本身是否应变得确定。

### Value loss 发散

reward scale/非平稳性大、value LR 高、终止 mask 错、bootstrap 跨 episode、value target 未 detach。做 reward/return normalization、value clipping、独立 LR 和正确 done mask。

### Advantage 异常

- 过大：reward outlier、value 失准；clip/normalize 并修 reward。
- 过小：RM 无区分、critic 拟合过强或 reward 全同。
- 全同：常见于组内答案/奖励一致，策略梯度接近零。

### 输出变长、重复

RM 长度偏好、逐 token KL/reward 聚合尺度不当、EOS 奖励不合理。按长度画 reward/胜率，做长度归一或数据去偏，而不是只加 max tokens。

## 7.7 PPO 手撕代码

```python
def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1)

def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
    loss = -masked_mean(torch.minimum(unclipped, clipped), mask)
    clip_frac = masked_mean((torch.abs(ratio - 1) > eps).float(), mask)
    return loss, clip_frac

def sampled_kl(new_logp, ref_logp, mask):
    return masked_mean(new_logp - ref_logp, mask)
```

一个 update step：固定 rollout 的 response、old log-prob、ref log-prob、reward；计算/保存 advantages 与 returns；多 epoch 按 minibatch 重算 current log-prob 和 value；优化 clipped policy loss 与 value loss；监控 reward、KL、entropy、ratio、clip fraction、grad norm；batch 完成后丢弃或限制复用并同步新 policy 给 rollout workers。

---

# 第八阶段：DPO 与离线偏好优化

## 8.1 从 KL-Regularized RL 到 DPO

对每个 prompt，标准目标：

$$
\max_\pi \mathbb E_{y\sim\pi(y|x)}[r(x,y)]
-\beta D_{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x))
$$

其最优策略满足：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)\exp(r(x,y)/\beta)
$$

反解 reward：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)
$$

把它代入 Bradley–Terry 的 `r_w-r_l`，只依赖 prompt 的 `log Z(x)` 抵消：

$$
P(y_w\succ y_l|x)=\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right)
$$

于是：

$$
L_{DPO}=-\log P(y_w\succ y_l|x)
$$

DPO 用策略相对 reference 的 log-ratio 直接参数化隐式 reward，因此无需显式 RM，也无需训练期间在线 rollout。

## 8.2 Sequence Log-Probability 如何算

$$
\log\pi(y|x)=\sum_{t=1}^{T}\log\pi(y_t|x,y_{<t})
$$

只 gather 实际 response token 的 log-prob，prompt 和 PAD 不计。必须正确右移 logits/labels。chosen/rejected prompt token 应一致，通常分别 forward 或在 batch 维拼接一次 forward。

按 token **求和**符合序列概率定义，但天然随长度累积；按长度平均改变了隐式 reward 定义。不同方法对此有不同修正，不能把 sum/mean 当无关实现细节。

## 8.3 `beta` 与 Reference 的作用

在常见 DPO loss 参数化中，β缩放相对 log-ratio margin。其与原始 RL 目标的 KL 系数、优化强度相关，但实际效果还受数据和学习率影响，不能简单说“β 越大 KL 一定越大/越小”而不说明公式约定。

reference 提供每个回答的基准难度：DPO 不是只提高 chosen 原始概率，而是提高 chosen **相对其 reference 概率的增幅**，并与 rejected 对比。常用 frozen SFT 模型；如果 ref 太弱或与数据来源差异太大，隐式奖励会失真。

## 8.4 DPO、SFT、PPO 对比

- SFT 只有 chosen，最大化其 token likelihood，不显式压 rejected。
- DPO 同时使用 chosen/rejected 和 frozen ref，做离线 pairwise 分类。
- PPO 用当前 policy 在线探索，显式/规则 reward 可优化未在离线 pairs 中出现的新回答，但系统更复杂不稳定。

DPO 的优势：简单稳定、两份 policy forward、无 RM/value/rollout；上限：只能从固定 pair 分布学习，无法主动探索当前 policy 的新行为，且容易受 preference noise/coverage 限制。

## 8.5 DPO 数据与训练问题

### 离线分布限制

若 pairs 来自远弱于当前模型的旧策略，任务太容易、风格不同；模型更新后进入新分布，数据没有对应偏好。解决：用目标模型附近的多策略生成、迭代刷新、按难度过滤。

### Length bias

序列 log-prob 是 token log-prob 求和，长度影响 margin；偏好标注本身也可能偏长。应报告 chosen/rejected 长度差与 reward margin 的相关性，做长度匹配切片，必要时选择带长度控制的目标。

### Pair 相似度

差距极大时 sigmoid 很快饱和、只学粗规则；近似 pair 梯度信息细但噪声高。可根据 judge 置信度、编辑距离、reward gap 分桶。

### Preference noise

错误 pair 会直接推动策略向错误方向。使用多人一致性、置信度加权、label smoothing/robust loss、去除高争议样本，并在独立 rubric 下复核。

### Loss 降而 win rate 不升

可能是对训练 pair 过拟合、reference log-prob/mask 实现错、评测 judge 偏差、数据覆盖不足、长度 shortcut、生成解码变化。至少同时看 held-out pair accuracy、隐式 reward margin、任务指标、人工/多 judge win rate 和长度。

隐式 reward 可定义：

$$
\hat r_\theta(x,y)=\beta(\log\pi_\theta(y|x)-\log\pi_{ref}(y|x))
$$

chosen reward accuracy 是 `r_chosen > r_rejected` 的比例；reward margin 是二者差。accuracy 高但 margin 极端可能意味着过拟合。

### Reference-free

实现上可把 reference log-ratio 当零或吸收进其他项，节省 ref forward；但也移除了显式行为锚点，可能更易漂移。它不是“没有任何先验”，初始化 policy 和数据本身仍是隐式先验。

## 8.6 DPO 变体的面试级理解

- IPO：认为严格 Bradley–Terry/logistic 假设与可分数据会导致过大 margin，使用回归式目标限制过度优化。
- KTO：可用单独的 desirable/undesirable 二元反馈，不要求同 prompt 成对回答，借鉴前景理论式价值函数。
- ORPO：在 chosen SFT NLL 上加入 chosen/rejected odds ratio 偏好项，单阶段训练且无需单独 ref。
- SimPO：使用平均 log-prob/长度归一化并引入目标 margin，通常不需 reference。
- RPO：名称在文献中可能指不同方法；面试必须先说明具体论文，不能只背缩写。
- CPO：同样存在多个全称/变体；回答时讲清它如何修改 loss、是否需要 ref、需要何种反馈。

## 8.7 DPO 手撕代码

```python
def response_logps(logits, labels, loss_mask):
    # logits [B,L,V]; token t 的 logits 预测 label t+1
    logp = logits[:, :-1].log_softmax(-1)
    target = labels[:, 1:]
    mask = loss_mask[:, 1:].bool()
    safe_target = target.masked_fill(~mask, 0)
    token_logp = logp.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (pi_chosen - ref_chosen).detach()
    rejected_reward = beta * (pi_rejected - ref_rejected).detach()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    margin = (chosen_reward - rejected_reward).mean()
    return loss, accuracy, margin
```

---

# 第九阶段：GRPO 与推理型 RL

## 9.1 GRPO 的核心思想

对同一个 prompt `x`，用 old/current-near policy 采样 `G` 个回答 `{y_i}`，得到奖励 `{r_i}`。用组内相对表现构造 baseline：

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,...,r_G)}
{\operatorname{std}(r_1,...,r_G)+\epsilon}
$$

然后把每个回答的 advantage 应用于其 response tokens，并使用 PPO 风格 ratio clipping 与 reference KL。直觉：不同 prompt 天生难度不同，直接比较绝对 reward 方差大；同 prompt 内比较“比同伴好多少”能消除 prompt 难度基线。

## 9.2 为什么可以不训练 Value Model

PPO 用 `V(s_t)` 预测状态 baseline。GRPO 使用同 prompt 多条 rollout 的经验均值（以及标准化）作为 baseline，不需额外 critic。优点是省一份模型、value optimizer 与不稳定的 value fitting；代价是每 prompt 必须多采样，baseline 粗、通常整条 response 共享序列 advantage，token credit assignment 仍弱。

严格地说，组内 mean 包含当前样本，会产生有限 group 的耦合/缩放；RLOO 使用“其他 G-1 个样本均值”作为 leave-one-out baseline，可减少 self-inclusion 影响。

## 9.3 Group Size 的影响

- 太小：mean/std 估计噪声大，容易所有 reward 相同。
- 增大：相对排名更稳定、探索覆盖更广。
- 代价：rollout token、KV Cache、reward 计算和显存/时间近似增加。
- 若 reward 二值且成功率很低，`G` 小时多数 group 全 0，没有梯度；可提高 G、调整任务难度/课程、改善探索或更密 reward。

当组内 reward 全同，分子为 0、std 为 0；加 epsilon 后 advantage 仍全 0，policy gradient 无学习信号，只有 KL 等项可能更新。

## 9.4 GRPO 的 Ratio、Clip 与 KL

对 token t：

$$
r_{i,t}(\theta)=\exp(\log\pi_\theta(y_{i,t}|s_{i,t})-
\log\pi_{old}(y_{i,t}|s_{i,t}))
$$

objective 与 PPO clipped surrogate 类似，只是 `A_i` 由组内 reward 得到，而非 critic/GAE。若多 epoch 重用 rollout，ratio 会逐渐偏离 1；clip fraction 高说明样本已太 off-policy 或更新过强。

Reference KL 抑制模型为获取规则 reward 而丢失语言质量。KL 既可作 reward shaping，也可在 loss 中显式加入；必须明确框架实现和符号。

## 9.5 GRPO、PPO、REINFORCE、RLOO

| 方法 | Baseline | Critic | 粒度 | 核心代价 |
|---|---|---:|---|---|
| REINFORCE | 可无/全局 | 否 | 常整轨迹 | 方差高 |
| RLOO | 同 prompt 其他样本均值 | 否 | 整轨迹 | 多 rollout |
| PPO | 学习的 V(s) + GAE | 是 | token/state | 多模型与 value fitting |
| GRPO | group mean/std | 否 | 常 sequence advantage 复制到 token | 多 rollout、组内偏差 |

不要说“GRPO 一定优于 PPO”。任务 reward 可验证、同 prompt 多采样可并行且 critic 难训练时 GRPO 很合适；需要细粒度长时序 credit 或 rollout 很贵时，critic/过程价值可能有优势。

## 9.6 奖励设计

### 四类奖励

- 格式：XML/JSON/推理与答案标签是否可解析。
- 正确性：最终答案、单元测试、仿真成功。
- 风格/安全：RM/judge 对表达与规范打分。
- 过程：每步证明、工具调用、驾驶行为阶段得分。

可验证奖励适合数学/代码，因为正确性可由解析器、测试或证明检查器低成本一致判断；开放写作/帮助性无法完全形式化，仍需 RM/人类。

### 稀疏奖励与长时序

终局二值奖励的信息密度低，长回答中无法定位关键错误。可以课程学习、提高采样、过程 reward、分阶段任务、搜索/重采样，但过程 reward 必须防止 shortcut。

### Reward hacking 示例

- 输出解析器可接受但语义错误的字符串。
- 硬编码测试样例、利用 sandbox/单测漏洞。
- 重复格式标签获取多次奖励。
- 从 prompt 泄漏答案或写“答案是……”骗 judge。

防护：严格 parser、隔离 hidden tests、reward 单次聚合、答案归一化、对抗测试、人工审计 top-reward failures。

### Reward scale、clip、normalize

不同 reward 分量量纲差异会使某项主导。先分别记录和标准化，再加权；reward clipping 抗 outlier，但会丢失强弱信息；group normalization 自动消除平移和尺度，但也让“全组都很差”的最好样本获得正 advantage，因此必须确保 sampling/课程合理。

### 长度与截断

若长回答更容易积累过程分或被 judge 偏爱，策略会变长；如果 max tokens 截断，正确长推理可能无最终答案得到 0，形成复杂偏差。要同时记录 generated length、finish reason、truncation rate、正确率随长度曲线，并在 reward 中显式区分完整性和冗余。

## 9.7 Curriculum 与 Process/Outcome 组合

课程学习可按模型当前成功率选择“非全会也非全不会”的 prompt，让 group 内有 reward 方差。过多简单题浪费 rollout；过难题全 0。动态 difficulty sampler 应基于历史成功率并保留固定分布防止遗忘。

过程和结果可组合：`R=αR_outcome+(1-α)R_process`，但不能只看总和；应分别监控，避免过程分掩盖最终错误。也可用过程分做 dense advantage，终局正确性做硬门槛。

## 9.8 Rollout 参数与 Pass@k

- temperature 高：多样性高、平均质量可能低、group reward variance 增大。
- top-p 小：截去尾部，降低灾难样本但可能限制新策略探索。
- max tokens：上限太短产生截断，太长增加成本并鼓励冗长。

如果每个独立样本成功概率为 `p`，至少一个成功：

$$
pass@k=1-(1-p)^k
$$

实际样本相关且常用无偏估计式，所以这是直觉而非通用精确计算。pass@k 提高可能只是多采样覆盖变广，pass@1 才更接近单次部署质量。

## 9.9 On-policy 成本、同步/异步与 Staleness

rollout 要逐 token decode，推理可能占 RL 大部分时间；生成长度不一又造成尾部拖慢。同步方案等整批 rollout 完成再更新，分布新鲜、实现简单，但 GPU 可能互相等待。异步让 rollout 与训练并行、吞吐高，但 worker 使用旧权重，产生 policy staleness。

staleness 后果：`π_old` 与当前 `πθ` 差异大，importance ratio 极端、clip 大量样本，梯度有效率低甚至偏差大。缓解：限制版本落后、频繁同步、丢弃太旧样本、单次小更新、importance correction/严格监控 KL 与 ratio。

## 9.10 GRPO 诊断

| 现象 | 可能原因 | 检查与处理 |
|---|---|---|
| reward 不涨/方差小 | 题太难/太易、温度低、reward 粗 | 看每组 unique/reward std；调课程、G、采样 |
| 长度持续增 | reward 长度偏差、截断设计 | reward-vs-length、truncation；去偏和长度约束 |
| KL 突升 | LR/epoch 大、旧 rollout、β小 | ratio/clip fraction/版本；减更新、增约束 |
| entropy 下降 | 探索塌缩、easy reward | 增采样多样性、降低更新、补多样数据 |
| 格式高正确率不升 | 格式 reward 太强 | 分项记录，正确性硬门槛/提高权重 |
| 不同 prompt 不稳 | 组小、二值稀疏、难度混合 | prompt-level success、动态课程、增 G |

## 9.11 GRPO 手撕代码

```python
def group_relative_advantage(rewards, group_ids, eps=1e-6):
    adv = torch.empty_like(rewards)
    for gid in group_ids.unique():
        idx = group_ids.eq(gid)
        r = rewards[idx]
        adv[idx] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv

def grpo_loss(new_logp, old_logp, seq_adv, token_mask, eps=0.2,
              ref_logp=None, kl_coef=0.0):
    # logp [B,T], seq_adv [B]
    ratio = (new_logp - old_logp).exp()
    adv = seq_adv[:, None]
    pg1 = ratio * adv
    pg2 = ratio.clamp(1 - eps, 1 + eps) * adv
    loss = -masked_mean(torch.minimum(pg1, pg2), token_mask)
    if ref_logp is not None and kl_coef > 0:
        loss = loss + kl_coef * masked_mean(new_logp - ref_logp, token_mask)
    return loss
```

完整 pipeline：prompt sampler 选 batch → rollout engine 按每 prompt G 条生成并保存 old log-probs/version → verifier/RM 打分并分项记录 → 组内 advantage → 训练 engine 重算 current/ref log-probs → clipped update → 监控 reward/KL/entropy/length/clip fraction → 同步权重并限制旧样本。

---

# 第十阶段：数据工程与模型评测

## 10.1 四类后训练数据不要混淆

| 数据 | 典型字段 | 用途 | 最大风险 |
|---|---|---|---|
| SFT | prompt + ideal response | 行为模仿 | 错答案被直接写入策略 |
| Preference/DPO | prompt + chosen + rejected | 离线偏好学习 | pair 噪声与分布过旧 |
| RM | 同上，可能含排序/标注者 | 学代理 reward | shortcut 与 OOD hacking |
| Rollout | prompt + sampled response + old logp + reward + version | 在线 RL | 数据快速过期、staleness |

DPO 数据与 RM 数据格式相近，但最佳分布和用途不完全相同；rollout 必须保留生成 policy 版本、采样参数和 token-level 信息，不能只存文本。

## 10.2 数据 Pipeline

### 清洗与格式校验

1. schema：字段存在、role 顺序合法、图像/视频路径可用。
2. 内容：空回答、乱码、重复、拒答模板、明显错误、PII/安全风险。
3. token：长度、EOS、截断率、不可见控制 token。
4. 一致性：answer 与 verifier、chosen/rejected 标签、图文对应。
5. 抽检：规则通过不代表语义正确，按数据源/难度分层人工抽检。

### 去重

- exact hash 处理完全相同文本。
- normalize 后 hash 处理空格、标点、大小写变化。
- MinHash/LSH 处理 n-gram 近似重复。
- embedding 相似度可查语义重复，但成本高且易误删合理相似题。

切分应按 prompt/来源文档/题族隔离，不能同一题的改写跨 train/test。

### 数据来源与许可证

记录 source、license、使用范围、抓取时间、是否含个人/机密信息、是否允许训练与再分发。面试时不要只说“网上收集”；给出数据 provenance 和删除/追溯机制。模型合成数据也可能继承源内容与许可证风险。

### Benchmark Contamination

训练中出现 benchmark 原题、答案、解析或近似改写，使分数反映记忆而非泛化。检测方法：exact/normalized match、n-gram/MinHash、embedding 检索、特殊短语搜索、时间切分；最可靠是维护私有/新鲜 holdout 和题目家族隔离。

### 版本管理与可复现

每个数据版本保存：原始 source snapshot、过滤代码 commit、配置、随机 seed、样本 ID、统计报告和产出 hash。训练 config 引用不可变数据版本；不能把“latest.jsonl”当可复现数据集。

## 10.3 数据分布与采样

至少统计：

- token 长度及截断率。
- 领域/任务/语言/安全类别。
- 难度、成功率、reward 分布。
- 来源模型与采样参数。
- chosen/rejected 长度差与相似度。
- 图像分辨率、视频帧数、动作长度等多模态属性。

若按原始数量混合，大源会淹没关键小源。常见策略：设置每源权重、temperature sampling、上/下采样、每 batch 配额、动态课程。重采样改变的是模型看到的训练分布，评测要回到真实目标分布。

### 合成数据的偏差与质控

强模型生成可快速扩规模，但会带来单一风格、错误自洽、模式复制、难度虚高/虚低和 teacher bias。质量控制：多 teacher/多采样、多 verifier 交叉、规则+人工抽检、保留生成 provenance、按 teacher 切片评测、去除模板重复。

### Rejection Sampling 与 Data Flywheel

对 prompt 采样 N 个候选，用 verifier/RM 选优形成 SFT/pairwise 数据；新模型再产生更难/更多样 rollout，继续筛选和训练，形成 flywheel。风险是 verifier 偏差被不断放大、模型只学会已有成功模式。需引入新任务、困难负例和独立人工评测。

### 主动学习与难例挖掘

优先标注模型最不确定、多个 judge 分歧、reward margin 小、线上失败或高 reward 失败的样本。它提高单位标注信息量，但采样分布不再代表真实流量，评测集不能照此偏置。

## 10.4 自动评测指标

### Accuracy、F1、EM、BLEU/ROUGE

- Accuracy：有唯一离散标签时直观；类别不均衡时可能误导。
- Precision/Recall/F1：检索、抽取、不均衡二分类；micro/macro 定义要明确。
- Exact Match：答案规范化后完全一致，适合短答案；同义正确答案可能被判错。
- BLEU：n-gram precision 为主，传统翻译；生成开放回答相关性有限。
- ROUGE：n-gram recall/LCS 为主，传统摘要；不保证事实正确。

### Pass@k

从 n 个候选中有 c 个通过 verifier，无偏估计常写：

$$
pass@k=1-\frac{\binom{n-c}{k}}{\binom nk}
$$

当 `n-c<k` 时为 1。比较必须固定采样温度、n、k、max tokens 和 verifier。pass@k 高但 pass@1 低，说明能力存在于分布尾部，单次部署未必好。

### Pointwise、Pairwise、Listwise Evaluation

- pointwise：judge 独立打分，容易受评分尺度漂移。
- pairwise：直接比较 A/B，通常稳定，但依赖对手和顺序。
- listwise：多模型排序，信息丰富但排序一致性难。

### Win Rate 与参考系

`win/(win+loss+tie)` 或把 tie 计 0.5 都有人使用，必须写清。win rate 不是绝对能力：换 baseline、prompt 分布、judge 或 decoding 都会变。多模型可用 Bradley–Terry/Elo 类模型估计相对强度，但仍依赖比较图连通和 judge 质量。

## 10.5 LLM-as-a-Judge

给 judge 明确 rubric、输入、候选答案，要求按维度或 pairwise 输出可解析结果。优点是便宜、可规模化；局限：

- position bias：偏第一个/第二个；交换顺序，两次不一致时记 tie/人工复核。
- verbosity bias：偏长；做长度匹配切片。
- self-preference：偏自身风格/家族；用多家 judge 或人工校准。
- style bias：标题、引用、措辞掩盖事实错误；把 correctness 与 style 分开。
- reference leakage：参考答案可能不完整，judge 盲从。

应在小规模人工集上验证 judge-human 一致性，固定版本和 prompt，并保存 judge 原始输出。

## 10.6 Benchmark 与真实产品效果为何不一致

- benchmark 分布与线上用户不同。
- 静态题目被污染或过拟合。
- 离线只测一次回答，线上是多轮、工具、延迟和安全约束。
- 解码、上下文长度、系统 prompt 不同。
- 产品价值是多目标：正确率、完成率、延迟、成本、投诉率等。

因此建立层级评测：单元/规则 → 固定 benchmark → 私有回归集 → 人工 pairwise → shadow/online A/B。高风险上线需要安全与回滚机制。

## 10.7 置信区间、Bootstrap、Seed

样本均值只是估计。Bootstrap 从评测样本有放回重采样，多次计算指标，用分位数得置信区间；pairwise 数据应按 prompt 为单位 resample，避免同 prompt 多回答泄漏独立性。

多个训练 seed 反映初始化、数据顺序、采样噪声。至少报告均值、标准差、seed 数和每个值；若训练成本不允许多 seed，应诚实说明，并用 checkpoint/评测 bootstrap 量化部分不确定性。

统计显著不等于业务显著：样本量大时微小差异也显著，仍需报告 effect size 和成本。

## 10.8 人工评测与错误分析

### Rubric

把“好”拆为 correctness、instruction following、completeness、relevance、clarity、safety；定义每个等级和边界例子。安全/正确性可设硬门槛，避免被文风总分抵消。

### 盲评与随机化

隐藏模型名，随机 A/B 顺序；同一 pair 交换后抽样复测。标注者不应看到训练 reward 或团队预期。

### 标注一致性

报告 raw agreement、Cohen's kappa（两人）或 Fleiss' kappa（多人）等；低一致性可能说明任务主观或 rubric 模糊，不应只把它当标注者质量差。

### Failure Taxonomy

先看案例，建立互斥尽量清晰的类别：知识错误、推理错误、指令遗漏、格式错误、幻觉、拒答不当、工具错误、长度/重复、安全问题。记录严重度和根因假设；每次模型更新看各类别迁移，而非只看总分。

### 固定回归集

包含历史严重失败、关键业务 case、边界与安全 case；不能拿它反复直接训练，否则又变训练集。可维护隐藏扩展集。

能力退化是任务本身不会做；格式退化是能力可能在但输出协议错；安全退化是违规/拒答边界改变。三者修复方法不同。

## 10.9 实验设计

### Baseline

至少包括当前生产/基座模型、只做 SFT 的模型、同算力/数据的合理替代方法。baseline 必须使用同一评测 prompt、模板、decoding 和预算。

### 控制变量与 Ablation

只改一个核心因素；若算法改动同时改变数据量和计算量，需额外做 compute/data matched baseline。消融优先验证“论文声称的因果组件”：去掉某 reward、改 group size、去 KL、改数据配比等。

### 实验记录

保存代码 commit、依赖/容器、模型与 tokenizer revision、数据 hash、完整 config、随机 seed、硬件、日志和 checkpoint。报告所有主要 seed 的均值/方差/样本量，避免 cherry-pick 最佳 checkpoint。

### Offline 与 Online 不一致

先对齐输入分布、prompt、decoding、延迟和工具配置；再看指标是否为代理、judge 是否偏、线上用户适应/反馈闭环。不要用线上结果倒推“离线评测没意义”，而应找到中间断点并加入回归集。

### 有限预算下选实验

先做便宜且能否定假设的实验：小模型/短程 run 验实现，固定小回归集测方向；再做关键超参数粗网格；最后才多 seed 和放大。选择信息增益最高的实验，而非把所有参数等距扫一遍。

### 数据—训练—评测闭环标准答案

定义任务与真实指标 → 建立不可污染的固定评测 → 对数据做来源、质量、长度和难度统计 → 训练最小 baseline → 监控训练与切片指标 → 做关键消融 → 对失败案例分类 → 把真实失败转化为新数据/奖励改进 → 用未参与训练的回归集和人工评测复验。

---

# 第十一阶段：训练系统、显存与推理基础

## 11.1 训练显存由什么组成

主要四项：

1. 参数 weights。
2. 梯度 gradients。
3. 优化器状态，如 Adam 的一阶/二阶矩及可能的 FP32 master weights。
4. activation：随 batch、sequence length、hidden/layers 增长，反向前需要保存。

另有 CUDA context、通信 buffer、allocator 碎片、临时 kernel workspace、logits 等。

### 基础估算

N 个参数仅存权重：

- FP32：`4N` bytes。
- FP16/BF16：`2N` bytes。
- 4-bit：理论 `0.5N` bytes，实际加 scale/metadata。

混合精度 Adam 的粗略训练 model states 常在每参数十几 bytes 量级，具体取决于是否保留 FP32 master、梯度 dtype、optimizer 实现；面试应拆项算，不背固定“16 bytes”而不说明假设。

LoRA 省的是可训练参数的梯度和 optimizer states；冻结基座仍需存权重，forward 也有 activation。QLoRA 再压缩冻结基座权重。

## 11.2 Mixed Precision、FP16 与 BF16

混合精度用低精度做大部分 matmul/存 activation，以 FP32 做敏感统计或 optimizer states。

| 格式 | 指数位 | 尾数位 | 特点 |
|---|---:|---:|---|
| FP16 | 5 | 10 | 精度较好但动态范围小，易 overflow/underflow |
| BF16 | 8 | 7 | 动态范围近 FP32，更稳，精度粒度较粗 |

FP16 常用 loss scaling：先把 loss 乘大再 backward，避免小梯度 underflow；step 前 unscale 并检查 overflow。BF16 动态范围大，通常不需要 loss scaling，但敏感 reduction/softmax/norm 仍常用 FP32。

## 11.3 Gradient Accumulation、Checkpointing、FlashAttention

- Gradient accumulation：多次 micro-batch backward 后一次 optimizer step，模拟大 effective batch；不减少单 micro-batch activation 峰值，且增加每次 update 的前反向次数。
- Activation checkpointing：forward 只保存部分节点，backward 重新计算其余 activation，省显存换算力；不要与保存训练 checkpoint 混淆。
- FlashAttention：对 Q/K/V 分块，在片上 SRAM 中做 online softmax 和局部累积，避免把完整 `L×L` attention matrix 反复写入 HBM。它是精确 attention（忽略浮点运算顺序差异），主要降低 IO 和中间内存，理论 attention FLOPs 的二次项仍在。

## 11.4 OOM 与吞吐低排查

### OOM

先确认是 model states、activation、KV Cache 还是碎片：

- batch/sequence 长导致 activation：减 micro-batch、checkpointing、packing、FlashAttention。
- 模型 states：LoRA/QLoRA、ZeRO/FSDP、offload。
- logits `[B,L,V]` 巨大：chunked CE/vocab parallel，避免保存不必要 logits。
- 瞬时峰值/碎片：看 memory snapshot、固定 shape、allocator config，不只看平均 allocated。

### 吞吐低

- GPU utilization 低：DataLoader/CPU tokenize、同步 barrier、batch 太小。
- 通信高：并行策略不匹配、跨节点带宽、频繁 AllGather。
- kernel 低效：shape 小/不对齐、未用 fused kernel/FlashAttention。
- 长尾：不同序列长度导致 padding 或 RL rollout 尾部等待。
- activation checkpointing/CPU offload 过度，省内存但计算/传输拖慢。

CPU/NVMe offload 把参数/optimizer/activation 移出 GPU，扩大可训练规模但受 PCIe/NVMe 带宽与延迟限制；它是容量方案，不是免费加速。

## 11.5 分布式并行

### Data Parallelism / DDP

每卡完整模型，处理不同数据；backward 对梯度 AllReduce，更新后权重一致。易用、扩展 batch，但每卡必须放下完整 model states。

### Tensor Parallelism

把单层矩阵按行/列切到多卡，单层计算需 AllReduce/AllGather。适合单模型放不下或追求单请求低延迟，要求高速互联，通信频繁。

### Pipeline Parallelism

按层切 stages，micro-batches 像流水线流过。起始/结束存在 bubble；更多 micro-batch 可降低 bubble 比例，但增加调度和 activation 管理复杂度。

### Sequence/Context Parallelism

把序列维切到设备。Sequence parallel 常分担 layernorm/dropout 等序列激活；context parallel 面向 attention 长上下文，需要交换 K/V 或采用 ring attention 等。两者具体定义依框架，回答时需说明所指实现。

### 3D Parallelism

组合 DP×TP×PP：TP 常在节点内高速互联，PP 跨模型层，DP 复制整个并行模型处理更多数据。最优映射取决于模型形状、网络拓扑、batch 和序列长度。

## 11.6 ZeRO-1/2/3 与 FSDP

传统 DP 每卡复制全部 model states。ZeRO 在 data-parallel ranks 间切分：

| Stage | 切分内容 | 每卡仍完整保存 |
|---|---|---|
| ZeRO-1 | optimizer states | 参数、梯度 |
| ZeRO-2 | optimizer states + gradients | 参数 |
| ZeRO-3 | optimizer states + gradients + parameters | 按需 gather 参数 |

ZeRO-3 forward/backward 前按层 AllGather 所需参数，之后释放/重分片，梯度用 ReduceScatter。显存显著下降，但通信和调度更复杂。

FSDP 与 ZeRO-3 思想相近：参数、梯度、optimizer states 全分片并按模块 materialize；API、调度和实现细节不同。不能说两者完全相同。

### Collectives

- AllReduce：各 rank 输入求和/均值，每 rank 得完整结果；DDP 梯度常用。
- ReduceScatter：求和后每 rank 只拿一片；分片梯度常用。
- AllGather：每 rank 的一片汇集成完整张量；ZeRO-3 参数 materialize 常用。
- All-to-All：每 rank 向所有 rank 发送不同分片；MoE token 路由常用。

## 11.7 MoE Expert Parallelism

把不同 experts 放不同设备。router 决定 token 去向；token hidden states 经 All-to-All 发到 expert 所在卡，expert 计算后再 All-to-All 返回。瓶颈包括负载不均、跨节点带宽、小 token batch 导致 GEMM 不饱和、capacity overflow。shared expert/辅助 loss/动态容量是常见手段。

## 11.8 推理：Prefill、Decode、Continuous Batching 与 PagedAttention

Prefill 一次处理 prompt，矩阵乘大、算力密集；decode 每步一个 token，要反复读模型权重和 KV，常带宽密集。

静态 batching 要等所有请求结束，短请求被长请求拖住。Continuous batching 在每个 decode step 移出已完成请求、加入新请求，提高利用率；调度器还需权衡 prefill 与 decode，防止长 prompt 阻塞已有请求。

PagedAttention 借鉴虚拟内存，把每个请求逻辑连续的 KV Cache 映射到固定大小物理 blocks：

- 减少预分配与内外碎片。
- 请求增长时按 block 分配。
- beam/prefix 可通过 block 表共享 KV，并用 copy-on-write。

它解决的是 KV 内存管理和调度，不改变 attention 的数学结果。

## 11.9 吞吐、延迟、TTFT、TPOT

- Throughput：每秒完成 requests 或生成 tokens；需说明是否含 prompt tokens。
- Latency：单请求端到端耗时。
- TTFT：Time To First Token，主要受排队和 prefill 影响。
- TPOT/ITL：首 token 后每个输出 token 时间，主要反映 decode。

提高 batch 往往提高吞吐但增加排队/单请求延迟，服务优化是 SLO 下最大化吞吐，不是只追 tokens/s。

vLLM 以 PagedAttention、高吞吐调度为核心；SGLang 更强调结构化生成/程序化接口、前缀复用与高效 runtime。项目回答应以实际使用版本的官方能力为准，不把两者简单说成“一个训练、一个推理”。

Prefix cache 保存相同 prompt 前缀的 KV，后续请求复用 prefill；需处理 hash、位置编码、模型/adapter 版本和缓存淘汰。Speculative decoding 用小 draft model 提议多个 token，大 model 并行验证；接受率高时减少大模型 decode 次数，输出分布可保持与目标采样一致。

## 11.10 RL 系统四部分

1. Rollout workers：高吞吐生成，保存 token、old logp、版本。
2. Reward workers/environment：RM forward、verifier、工具/仿真。
3. Training workers：前反向、optimizer、分布式训练。
4. Orchestrator/data store：调度 prompt、轨迹、版本、容错和监控。

### Colocated 与 Disaggregated

- Colocated：同一 GPU 集群时间复用 rollout 和 training；资源利用灵活、少跨集群传输，但频繁切换/重分片、显存竞争。
- Disaggregated：独立生成池和训练池；可各自选择推理/训练并行，流水并发，但要传轨迹和同步大权重，staleness 更明显。

训推分离的根本原因：推理追求小延迟/高并发、KV 管理和 TP；训练追求大 batch、activation/optimizer、DP/ZeRO。最优布局不同。

### 同步与异步

- 同步：rollout 一批 → reward → update；on-policy 程度高但存在 straggler 和资源空闲。
- 异步：生成、reward、训练并行流动；吞吐高但样本旧、版本管理和 correction 复杂。

### 动态批处理与长短不均

按剩余长度/预计长度 bucket，continuous batching，chunked prefill，限制单请求预算；训练端可按总 tokens 组成 batch。RL 最后几条超长 rollout 会造成 tail latency，可设置合理 max length 但要记录截断偏差。

### 权重同步与重分片

训练可能用 ZeRO/FSDP，推理用 TP；同步时需把分片参数 gather、转换成推理布局再广播。频繁同步保证新鲜却成本高，稀疏同步提高吞吐却增加 staleness。LoRA 训练时可只传 adapter，但 rollout 合并/挂载也需版本一致。

Ray 可做分布式 actor、资源 placement 与容错编排；Kubernetes 管 pod、资源、重启和集群调度。它们解决系统编排，不替代 PPO/GRPO 算法。

## 11.11 系统指标与联合监控

- tokens/s、samples/s：同时报告 prompt/generated/effective trained tokens。
- GPU utilization：粗指标；高利用率不保证有效算力高，可能在做重算/无效 padding。
- MFU：实际有效模型 FLOPs ÷ 理论峰值 FLOPs；依赖 FLOPs 估算和硬件精度，跨实现需谨慎。
- rollout acceptance/success、平均长度、截断率、KV 使用率。
- data loading、compute、collective、idle、reward、weight sync 的 wall-time breakdown。

“生成吞吐”是 rollout engine 每秒生成 token；“有效训练吞吐”应扣除丢弃旧样本、padding、无 advantage group、过度 clipped tokens。算法效率与系统吞吐必须联合看：tokens/s 提高但有效梯度比例下降，不一定更快收敛。

标准 dashboard 同时展示：reward 分项、pass rate、KL、entropy、advantage/reward std、长度、ratio/clip fraction、grad norm；以及 tokens/s、GPU 利用、KV 占用、通信、队列、权重版本差和故障率。

---

# 第十二阶段：多模态大模型专项

## 12.1 CNN 与 ViT

CNN 通过局部卷积、权重共享和平移等变引入强视觉先验，样本效率好；ViT 把图像变成 patch token，用全局 self-attention 建模远程关系，结构统一、易与 Transformer/LLM 连接，但通常更依赖大规模预训练。

给图像 `H×W×C`、patch `P×P`，patch 数：

$$
N=\frac HP\frac WP
$$

每个 patch flatten 成 `P²C`，经线性层映到 `d`。加可学习/固定位置编码；分类 ViT 可在开头加入 `[CLS]`，最终 hidden 作分类，也可做 global average pooling。VLM 通常保留 patch tokens，不只取 CLS。

分辨率从 H/W 各扩大 2 倍，token 数约增 4 倍，标准 self-attention 二次项可能增约 16 倍；这解释高分辨率 VLM 为什么常用 tiling、局部 attention、token pruning/resampler。

## 12.2 CLIP 图文对比学习

图像编码器得归一化向量 `v_i`，文本编码器得 `t_j`，相似度：

$$
s_{ij}=\frac{v_i^Tt_j}{\tau}
$$

batch 内正确图文为对角线，分别做 image-to-text 和 text-to-image CE：提高匹配 pair 相似度、降低其他 pair。CLIP 学到共享语义空间，可用类别文本 prompt 与图像相似度做 zero-shot 分类。

局限：batch negatives 可能含语义等价 false negatives；对细粒度空间、计数、OCR 未必强；全局对比目标不天然提供 region grounding。

## 12.3 Vision Encoder + Connector + LLM

典型 VLM：

1. Vision encoder（常为 CLIP/SigLIP ViT）输出视觉 tokens。
2. Connector 把视觉 hidden size/数量转换到 LLM embedding 空间。
3. 视觉 tokens 与文本 token embeddings 组成统一序列，LLM 自回归回答。

### Connector 对比

| 方式 | 机制 | 优点 | 局限 |
|---|---|---|---|
| Linear projector | 每个视觉 token 线性映射 | 简单、参数少 | 跨模态交互能力有限，不压 token |
| MLP projector | 加非线性变换 | 容量更强 | 仍常一对一保留 token |
| Q-Former/Resampler | 固定 learnable queries cross-attend 视觉特征 | 压成固定 token 数、选择信息 | 额外模块/预训练，可能丢细节 |
| LLM 内 cross-attention | 文本层显式读取视觉 memory | 深度融合、可按文本查询 | 改 LLM 结构和训练/部署复杂 |

### Q-Former 具体在做什么

设视觉特征 `V∈R^{N×d_v}`，M 个可学习 queries `Q∈R^{M×d_q}`。Q-Former 的 query self-attention 让 queries 交流，cross-attention 用 query 作 Q、视觉特征作 K/V：

$$
Q'=\operatorname{softmax}\left(\frac{(QW_Q)(VW_K)^T}{\sqrt d}\right)VW_V
$$

不论原图有多少 patch，输出固定 M 个视觉摘要 token。learnable queries 是训练参数，不是每张图永久更新的 memory：训练时 SGD 更新全局 query 参数；推理时参数固定，但每张图产生不同 query hidden states。

## 12.4 视觉 Token 如何进入 LLM

常见做法是把 `<image>` 占位符替换成投影后的视觉 embeddings：

```text
<system> ...
<user> <image_start> v1 v2 ... vN <image_end> 问题
<assistant> 回答
```

LLM causal attention 让回答 tokens 读取前面的视觉 tokens。若视觉 tokens 在序列中间，要正确构造 position IDs、attention mask 和 labels；视觉位置通常不计算语言 CE。

多图可为每张图设置边界 token 并拼接；视频常按帧采样，每帧 patch tokens 再加时间位置/帧标识，或先用时空 encoder/resampler 压缩。直接把全部帧拼接会快速耗尽 context。

## 12.5 冻结/解冻策略

| 策略 | 优点 | 风险/适用 |
|---|---|---|
| 冻结 vision+LLM，只训 connector | 便宜稳，适合初始对齐 | 上限受现有表征限制 |
| 冻结 vision，训 connector+LLM/LoRA | 提升指令和跨模态使用 | 可能损伤语言能力 |
| 解冻 vision 部分/全部 | 适应专业图像、细粒度任务 | 成本高、小数据易过拟合 |
| 全量端到端 | 最大容量 | 数据/算力要求最高、遗忘风险 |

模态对齐和语言保留是权衡：强推视觉任务可能改变 LLM；可混入纯文本 replay、低 LR/LoRA、分阶段解冻、分别评测 text-only 与 vision-language 能力。

## 12.6 VLM 三阶段训练

1. 图文/特征对齐：大量 image-caption pair，常冻结两端、训练 connector，使视觉可被 LLM 表示。
2. Multimodal SFT：图像+指令+回答，学习问答、OCR、grounding 和多轮交互。
3. Preference/RL：同一图文 prompt 的 chosen/rejected，或可验证视觉任务 reward，优化幻觉、格式、推理与行为。

数据差异：caption pair 描述图像，不一定是指令；instruction data 有交互意图和理想回答；preference data 比较多个回答，常需控制图像与 prompt 相同。

Multimodal chat template 要明确每张图对应哪个占位符；labels 只监督 assistant 文本或 action tokens。batch 中不同尺寸可 resize/pad、动态分辨率/tiling；不同帧数可采样、padding + frame mask、token budget 分配。padding 不仅存在文本维，也存在视觉 token/帧维。

## 12.7 高分辨率策略

- 直接 resize：token 固定、快，但小字/细节丢失、几何变形。
- letterbox/pad：保比例但有空白。
- global thumbnail + local tiles：保全局与局部细节，token 数增加。
- dynamic resolution：根据长宽比分块，batch shape 复杂。
- resampler/token pruning：压缩 token，可能丢 OCR/计数细节。

分辨率提高不保证效果线性提高；需报告视觉 token 数、计算/显存、OCR/细粒度切片和通用指标。

## 12.8 视觉幻觉、OCR、计数、空间与细粒度问题

视觉幻觉是模型生成图中不存在或证据不足的对象/属性/关系。来源：语言 prior 强于视觉证据、connector 信息瓶颈、低分辨率、训练 caption 偏置、teacher 幻觉、评测诱导。

- OCR：需高分辨率、文本丰富数据、可能专用 encoder；切块会破坏阅读顺序。
- 计数：global pooled 表示和语言模式难保实例级对应，需要高分辨率/region 表示/检测辅助。
- 空间关系：2D position 在压缩/拼接后可能弱化，需 grounding/坐标监督。
- 细粒度识别：视觉 encoder 的预训练类别/分辨率限制。

缓解：更好视觉 backbone/分辨率、grounded data、负例与拒答、区域/坐标 token、视觉 verifier、多模态 preference/RL；同时要求模型在证据不足时表达不确定性。

## 12.9 多模态评测

- 避免仅靠 OCR 后的文本泄漏或题库污染。
- 按感知、OCR、空间、计数、知识、推理、幻觉分桶。
- 人工 rubric 分“看对了什么”和“推理/回答是否对”，定位 vision/LLM 哪端失败。
- 对同一问题做图像替换、遮挡、反事实，测试是否真的依赖视觉。
- 多模态 RM 输入通常为图像+prompt+response，架构可与 VLM 相同加标量 head；需防止仅看文本风格打分。

Grounding 把文本概念与 region/bounding box 对齐；坐标可离散为 tokens、连续回归或用 region features。region token 让 LLM 引用局部视觉表示。分辨率提高的成本需要与 grounding 精度共同报告。

## 12.10 VLA / 自动驾驶建模

VLA 学：

$$
\pi(a_{t:t+H}\mid o_{\le t},\text{instruction/history})
$$

observation 可为多相机图像、视频、状态；language 提供任务/解释/高层意图；action 是控制或轨迹。

### Action 表示

| 表示 | 优点 | 缺点 |
|---|---|---|
| 连续动作回归 | 精细、直接控制 | 多峰分布下均值化；需专用 head/loss |
| 离散 action tokens | 与 LLM next-token 统一，易做 CE/RL | 量化误差、词表/顺序设计 |
| Trajectory/waypoints | 平滑、规划语义强，可下游控制 | 仍需 controller，闭环误差 |
| Diffusion/action chunk | 表达多峰连续分布、一次出序列 | 采样/训练更复杂、延迟 |

Action chunking 一次预测未来 H 步，减少逐步推理延迟并保持短期一致性；H 太长又降低对新观测的反馈频率。可滚动执行前若干步再重规划。

## 12.11 Behavior Cloning、Imitation Learning 与 RL

Behavior Cloning 用 expert `(o,a)` 做监督 MLE/MSE，是模仿学习的一种。广义 imitation 还包括 DAgger、逆强化学习等。RL 从 reward 与环境交互优化，不要求每步 expert action，但样本昂贵且安全风险高。

### Covariate Shift 与 Compounding Error

BC 只在 expert state 分布训练；部署中一个小错误进入未见状态，后续错误继续累积。若每步错误率 `ε`，长期成本可能远超独立 `Tε` 的简单直觉。DAgger 让当前 policy rollout，再由 expert 标注其访问状态，逐轮缩小分布差异。

## 12.12 Open-loop 与 Closed-loop

- Open-loop：固定日志输入，预测动作/轨迹与 recorded GT 比较；快、可复现，但 GT 只是一个可行解，不能反映动作改变未来观测。
- Closed-loop：模型动作影响仿真/真实环境后续状态，测碰撞、到达、舒适、规则和恢复能力；更接近部署但仿真真实性、方差和成本高。

离线 L2/轨迹误差下降不保证闭环更好，因为小误差可能在关键场景导致碰撞，或预测不同但同样安全的轨迹被惩罚；模型还会进入日志中没有的状态。

## 12.13 驾驶 Reward 多目标冲突

- 安全：碰撞、TTC、越界，通常应为硬约束或极高权重。
- 规则：红灯、限速、路权。
- 进度/到达率：避免模型为安全完全不动。
- 舒适：加速度、jerk、横摆。
- 效率：时间、路径长度。

简单加权和会出现尺度和 trade-off：提高进度可能牺牲舒适/安全。可用约束 RL、层级优先级、Pareto 分析或硬规则+软 reward；始终分别报告各分项，不能只报告总 reward。

长时序中终局碰撞/到达稀疏，credit assignment 困难。可设计 progress、TTC、lane keeping 等 dense shaping，但 shaping 可能让模型钻局部指标；最终仍需闭环安全 case 验证。

## 12.14 VLA、世界模型与传统模块化规划

- 模块化：感知→预测→规划→控制，接口清晰、可解释、便于安全约束，但误差层层传递、联合优化难。
- VLA：统一感知/语言/动作，可利用大规模多模态知识和端到端优化，但可解释/验证、实时性和闭环数据要求高。
- 世界模型：预测环境未来状态/潜变量，可用于想象 rollout、规划或训练策略；它不是 VLA 的反义词，VLA 可以内部或外部使用世界模型。

## 12.15 历史记忆、流式推理与数据闭环

历史记忆处理遮挡、速度和长期意图。直接拼全历史 token 成本增长；可滑窗、KV Cache、memory tokens、Q-Former/resampler 压缩。所谓流式推理：观测连续到达，模型复用过去状态/缓存并增量更新，而不是每帧从零处理全历史。要处理 memory drift、重置、时间对齐和训练—推理一致性。

数据闭环：线上/仿真日志 → 规则与模型挖掘困难/失败场景 → 去重和场景聚类 → 人工/自动标注 → 训练 → 固定回归+闭环验证 → 再部署。困难场景不能只按模型高 loss 选，还应覆盖安全严重度、罕见性和场景多样性。

### 多模态/VLA 阶段验收标准答案

典型 VLM：冻结或微调 ViT 视觉编码器，connector 将 patch features 投到 LLM hidden space，视觉 tokens 插入 chat sequence，LLM 对 assistant tokens 做 causal loss；先对齐 projector，再 multimodal SFT，最后用偏好或可验证 reward 对齐。VLA 进一步把输出替换/增加 action tokens、连续 action head 或 trajectory head，并必须做闭环评测。

相对纯文本，多模态后训练新增图文/时序对齐、分辨率和视觉 token 预算、图像对应关系、视觉幻觉、模态专属 reward，以及“语言模型是否真正使用视觉”的反事实评测。VLA 还新增动作执行后改变未来状态的闭环分布问题。

---

# 第十三阶段：项目追问与科研表达

这一阶段没有统一“标准事实答案”，但有统一的证据结构。面试官不是只看结果，而是在判断：你是否真正做过、能否把问题建模、能否区分因果和相关、遇到失败能否定位。

## 13.1 30 秒与 2 分钟项目介绍

### 30 秒结构

> 我针对【场景/用户】中的【具体问题】，基于【模型和数据规模】设计了【SFT/DPO/GRPO/VLA 等方法】。我的核心贡献是【数据、算法或系统的可归因改动】；在【固定评测/闭环指标】上，相比【合理 baseline】提升【数值】，同时【通用能力/安全/成本】变化为【数值】。我还通过【消融或失败分析】确认收益主要来自【组件】。

不能只说“做了一个大模型微调项目，效果不错”。数字必须说明评测集、样本量、是否多 seed，以及提升是绝对值还是相对百分比。

### 2 分钟 STAR+Research 结构

1. **背景/任务**：为什么重要，输入输出是什么，真实约束是什么。
2. **难点/假设**：现有 baseline 失败在哪里，你的可验证假设是什么。
3. **方法**：数据、模型、目标函数、训练系统，重点讲自己的改动。
4. **实验**：baseline、公平设置、主结果、消融、失败切片。
5. **结论/局限**：确认了什么，不能证明什么，下一步是什么。

## 13.2 Pipeline 怎么画

项目 pipeline 应把训练数据流和推理数据流区分开。例如后训练项目：

```text
原始数据 → 清洗/去重/切分 → SFT → baseline 评测
                              ↓
prompt → 多样采样 → verifier/judge → preference/reward
                              ↓
                         DPO/GRPO 更新
                              ↓
              通用+领域+安全+效率评测 → 错误分析
```

图上标出：哪些模块冻结/训练、模型版本、数据量、关键 loss、评测入口。VLA 再画 observation/action/environment 闭环，避免只画 open-loop 数据箭头。

## 13.3 如何说个人贡献

把“我们做了”拆成：

- 我独立负责：具体代码/实验/数据/设计。
- 我与他人共同：接口和分工。
- 已有框架提供：Trainer、基础模型、开源 baseline。
- 我的决策影响：做了什么比较，为什么选择最终方案。

可信表达示例：

> 训练框架基于开源实现，我主要负责数据 schema、assistant mask、两项 reward、GRPO 配置和评测。为了确认不是框架默认配置带来的收益，我复现同 compute SFT 与 DPO baseline，并手写核对了 log-prob 和 loss。

## 13.4 最重要技术决策怎么回答

使用“候选方案—约束—证据”结构：

> 当时有 A/B 两种方案。因为我们的瓶颈是【算力/数据/闭环反馈】而不是【另一个因素】，我选择 B。小规模对照显示【指标】，同时 B 的成本【数值】。代价是【局限】，所以增加了【回归/保护】。

如果无法说出被放弃的替代方案和证据，往往说明不是你做的决策。

## 13.5 算力、训练时长、模型与数据量

至少能回答：GPU 型号与卡数、模型参数/可训练参数、precision、最大长度、global batch/token batch、训练 steps/epochs、耗时、峰值显存、tokens/s、数据样本/token 数、rollout 数与平均长度。RL 还需 policy/ref/reward/value 的布局及生成/训练时间占比。

不要把“8 卡训练 2 天”当完整算力说明；卡型和 token 总量决定信息量。

## 13.6 数据追问模板

### 数据从哪里来，是否合法、是否泄漏

回答包括 source/provenance、license/隐私、采集时间、benchmark 去污染、train/val/test 按 prompt/题族隔离。若无严格许可证审计，要直接承认研究用途限制。

### 如何清洗、去重、过滤、采样

按规则→模型→人工抽检三层说明，并给每步保留率；说出近似去重方法、异常率和分层统计。采样要说明为什么该配比匹配目标流量/能力短板。

### 为什么使用这些配比

不能回答“经验设置”。应说目标分布、源数据规模差、关键小类、消融结果和通用回归。例如领域:通用从 8:2 调到 6:4 后领域只少 0.3，但通用恢复 2.1，因此选择 6:4。

### 最常见噪声

按项目实际讲，如错误答案、role 错位、模板残留、chosen/rejected 反标、verifier 解析失败、图文错配、自动驾驶时间不同步。说明发现方式和残余风险。

### Chosen/Rejected 或 Reward 如何构造

说明候选来自哪些 policy/temperature、谁打分、rubric、position randomization、置信度、hard negatives、冲突处理，以及是否接近目标 policy 分布。

### 数据量翻倍会怎样

正确答案不是必然变好。若只是重复相同分布，边际收益小；若补齐长尾/困难/当前 policy 数据，可能有效。算力固定时更多数据意味着每样本 epoch 降低。提出 scaling 小实验而非拍脑袋预测。

## 13.7 方法追问模板

### 为什么选 SFT/DPO/PPO/GRPO

- 有高质量理想答案、主要学格式/任务映射：SFT。
- 有离线 chosen/rejected、算力有限、无需探索：DPO。
- reward 可在线计算、需要探索新解、且可承担 critic/system：PPO。
- 同 prompt 易多采样、reward 可验证、想省 value model：GRPO/RLOO。

选法必须联系数据与约束，而非“GRPO 最新”。

### 目标函数与任务指标如何对应

解释代理链：SFT CE 提高示范 token likelihood，但最终关心任务正确率；DPO 提高偏好胜率；RL 直接用 verifier reward 更接近目标。说明代理缺口，并用独立任务指标防止 reward hacking。

### 关键超参数

SFT：LR、effective batch、epoch、max length、rank/alpha、数据配比。DPO：β、LR、pair 难度、sum/mean logp。PPO：clip ε、KL β/target、GAE γ/λ、update epochs、value coef。GRPO：group size、temperature、reward weights、clip、KL、max length、policy version lag。

### 哪些模块训练/冻结

逐模块说 vision encoder、connector、LLM/LoRA、LM/action/value/reward head。冻结不等于无 activation；如果冻结模块输入不需梯度，可减少图保存。

### 不使用 Trainer 如何实现

能写出 logits→response log-prob→mask→loss；PPO/GRPO 还要保存 old logp、计算 advantage、clip；DPO 要 current/ref chosen/rejected 四个 sequence logp。本文代码章节即最小答案。

### 方法假设和局限

SFT 假设 demonstrations 足够覆盖；DPO 假设偏好模型和离线 pair 有效；PPO 假设 RM 是可靠代理、on-policy 近似；GRPO 假设组内比较有 reward 方差且多采样可承受。主动说局限比宣称方法万能更可信。

## 13.8 实验追问模板

### Baseline 为什么合理

与当前方法同模型、数据、模板、token budget、decoding 和评测，只改变核心因素；包含当前强实践而非弱模型。若 compute 不同，做 compute-matched 对照。

### 是否控制变量

列出固定项，说明唯一改动；如果现实上改了多个因素，拆消融，不能硬说控制了。

### 最重要 Ablation

选择直接验证核心机制的一项。例如多意图 RL 的核心主张是意图多样性，就对比相同 rollout/token 预算下多意图 vs 单意图，而不只对比不同 seed 数量。

### 多 Seed 和方差

报告每个 seed、均值±标准差；若资源不足，说明只跑 1 seed，并用 bootstrap/重复评测与中间 checkpoint 作为有限稳定性证据，不伪装成确定结论。

### 指标是否有统计意义

给样本量、置信区间、paired test/bootstrap 和 effect size。模型在同 prompts 上比较时用 paired 分析更有力。

### 最失败实验

使用：预期 → 观察 → 假设 → 定位 → 修复/结论。示例：预期增大 group size 提升探索，实际 reward 不变且吞吐降；发现多数 prompt 全成功，组内 advantage 为零；改难度采样后有效。失败必须改变了你的理解。

### 三个失败案例

不要挑三个随机例子；至少覆盖三种根因，例如数据错误、reward shortcut、分布外/闭环失败，并说明数量占比和对应修复。

### 多一倍算力做什么

优先解决当前最大不确定性，不是简单模型翻倍。可能是更多 online rollout、多 seed、闭环仿真、困难 reward 数据或更完整 ablation；说明预期信息增益。

## 13.9 工程追问模板

### OOM、NaN、吞吐、稳定性

讲真实症状、监控、根因和量化修复。例如：长样本导致 activation 峰值；按长度 bucket + FlashAttention + checkpointing 后峰值从 X 降到 Y、吞吐变化 Z。不要只列“用了 DeepSpeed”。

### 监控什么

SFT：train/val loss、task metric、grad norm、LR、tokens/s、长度。DPO：loss、chosen/rejected logp、implicit reward accuracy/margin、KL/长度。RL：reward 分项、KL、entropy、adv/reward std、ratio、clip fraction、长度、成功率、staleness 和系统利用率。

### Checkpoint 与断点续训

保存 model/adapter、optimizer、scheduler、scaler、global step/epoch、sampler/RNG states、数据版本和 RL policy/version/queue 边界。只保存权重不能精确恢复训练轨迹。分布式 checkpoint 要防部分 rank 写坏，使用原子发布和完成标记。

### 可复现

固定 seed 仍不保证 GPU 完全确定；记录 CUDA/cuDNN/kernel、硬件和非确定算子。目标通常是指标在合理误差内可复现，而非 bitwise identical。

### 扩展到多机多卡

先判瓶颈：模型放不下用 ZeRO/FSDP/TP，吞吐不足用 DP，层太深/单模型巨大用 PP，长上下文用 CP。把高速通信放节点内，跨节点减少高频 collectives；处理 checkpoint、故障、数据 shard 和网络拓扑。

## 13.10 项目表达验收

一个可经 30 分钟追问的项目，应有：一张 pipeline、一个公平 baseline、主结果、至少一项核心消融、三类失败、一个系统问题、明确个人贡献和一个诚实局限。每个“提升”都能指向表格或案例，每个技术词都能说明为何使用。

---

# 第十四阶段：高频对比题速答

## 14.1 训练与对齐方法

| 对比 | 一句话抓手 | 关键细节 |
|---|---|---|
| Pretraining vs Continued PT vs SFT | 通用原始分布 vs 领域原始分布 vs 指令示范 | 前两者通常全 token LM loss；SFT 常 assistant-only |
| SFT vs DPO | 模仿 chosen vs 拉开 chosen/rejected 相对 ref 的 margin | DPO 需要 preference pair 和 ref |
| DPO vs PPO | 离线分类式偏好优化 vs 在线 rollout 最大化 reward | DPO 简单但不探索；PPO 复杂但可在线发现新行为 |
| PPO vs GRPO | 学 critic/GAE vs group-relative baseline | GRPO 省 value model但需同 prompt 多采样 |
| Reward Model vs Verifiable Reward | 学得的主观代理 vs 规则/环境客观反馈 | RM 覆盖开放偏好但可被 hack；verifier 可靠但覆盖窄 |
| Outcome vs Process Reward | 只看最终结果 vs 给中间步骤反馈 | outcome 客观稀疏；process 稠密但标注偏差大 |
| On-policy vs Off-policy | 当前策略数据 vs 可复用其他/旧策略数据 | 前者分布匹配但贵；后者高效但需 correction |

## 14.2 三个 Policy 名称

| 名称 | 更新？ | 时间尺度 | 目的 |
|---|---:|---|---|
| Current policy | 是 | 每个 optimizer step 变化 | 正在学习 |
| Old policy | batch 内冻结 | 每轮 rollout/update 刷新 | PPO ratio 的采样分布 |
| Reference policy | 通常否 | 整个 RL/DPO 固定 | 行为锚点/KL 基准 |

## 14.3 架构与数值

| 对比 | 答案 |
|---|---|
| MHA/GQA/MQA | KV heads 分别为 h/g/1；越少 cache 越省，表达可能下降。 |
| LayerNorm/RMSNorm | LN 减均值再除标准差；RMSNorm 只按 RMS 缩放，更简洁。 |
| Adam/AdamW | Adam 中 L2 会被自适应缩放；AdamW 把统一参数衰减与梯度更新解耦。 |
| FP16/BF16 | FP16 尾数多但指数少，范围小；BF16 范围近 FP32，训练通常更稳。 |
| Full FT/LoRA/QLoRA | 全参更新 / 冻结基座训低秩 / 4-bit 冻结基座训低秩。 |
| Dense/MoE | 每 token 激活全部 FFN / 路由到少数 experts；MoE 参数大但通信复杂。 |

## 14.4 分布式与推理

| 对比 | 答案 |
|---|---|
| DDP vs ZeRO/FSDP | DDP 每卡完整 model states、同步梯度；后者切分 states 省显存。 |
| DP/TP/PP/CP | 切数据 / 单层张量 / 网络层 / 序列上下文。 |
| Prefill vs Decode | 完整 prompt 大矩阵并行、compute-heavy / 单 token 迭代、KV/带宽/调度 heavy。 |
| Sync vs Async RL | 数据新鲜、等待多 / 流水吞吐高、staleness 大。 |

## 14.5 评测与多模态

| 对比 | 答案 |
|---|---|
| Offline vs Online | 固定数据快且可复现 / 真实交互更近部署但贵、有风险。 |
| Open-loop vs Closed-loop | 动作不改变未来输入 / 动作影响环境未来；驾驶部署必须关注后者。 |
| Encoder-only / Enc-Dec / Decoder-only | 双向理解 / 条件生成分两塔 / 因果统一生成。 |
| Linear Projector vs Q-Former | 一对一映射视觉 token / learnable queries cross-attend 并压缩选择视觉信息。 |

---

# 第十五阶段：最终手撕代码与白板索引

前文已经给出 Cross Entropy、RMSNorm、Attention、LoRA、SFT loss/mask、response log-prob、DPO、REINFORCE、PPO、GRPO、top-k/top-p 和 GAE。这里补齐 KV Cache 与多轮 collator，并给出面试书写顺序。

## 15.1 简化版 KV Cache Attention

```python
class CachedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def forward(self, x, cache=None):
        # prefill: x=[B,L,D], cache=None; decode: x=[B,1,D]
        q, k_new, v_new = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        if cache is not None:
            k = torch.cat([cache["k"], k_new], dim=-2)
            v = torch.cat([cache["v"], v_new], dim=-2)
        else:
            k, v = k_new, v_new

        score = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        # prefill 需要 causal mask；单 token decode 时所有 cache 都在过去，无需屏蔽
        if q.size(-2) > 1:
            Lq, Lk = q.size(-2), k.size(-2)
            mask = torch.triu(torch.ones(Lq, Lk, device=x.device, dtype=torch.bool), 1)
            score = score.masked_fill(mask, float("-inf"))
        y = score.float().softmax(-1).to(x.dtype) @ v
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.o(y), {"k": k_new if cache is None else k,
                           "v": v_new if cache is None else v}
```

生产实现会预分配/分页写入 cache，避免每步 `torch.cat` 的复制，并正确处理 RoPE position、GQA、batch 中不同长度。

## 15.2 多轮对话 Collator 的关键结构

```python
def encode_chat(messages, tokenizer, template):
    """template 返回整段 token ids，以及每个 assistant 内容的 token span。"""
    input_ids, assistant_spans = template(messages, tokenizer)
    labels = [-100] * len(input_ids)
    for start, end in assistant_spans:
        labels[start:end] = input_ids[start:end]
    return {"input_ids": input_ids, "labels": labels}

def chat_collator(features, pad_id):
    max_len = max(len(x["input_ids"]) for x in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for x in features:
        n = max_len - len(x["input_ids"])
        batch["input_ids"].append(x["input_ids"] + [pad_id] * n)
        batch["labels"].append(x["labels"] + [-100] * n)
        batch["attention_mask"].append([1] * len(x["input_ids"]) + [0] * n)
    return {k: torch.tensor(v) for k, v in batch.items()}
```

## 15.3 代码题推荐书写顺序

1. 先写输入 shape 和 mask 语义。
2. 只写核心正确实现，再补数值稳定和 corner cases。
3. 对 logits/labels 明确 shift。
4. 对 response logp 明确 prompt/PAD mask。
5. 对 RL loss 明确哪些张量 `detach`、哪些来自 old/ref。
6. 最后说生产优化：FlashAttention、fused CE、no_sync、paged cache 等。

## 15.4 必须能在 20 分钟内写出的代码

- Cross Entropy：`logsumexp - target logit`。
- RMSNorm：FP32 均方根、乘 scale。
- MHA：QKV reshape、score、mask、softmax、合头。
- LoRA：冻结 base，`B(A(x))*alpha/r`，B 零初始化。
- SFT：assistant-only labels + causal shift。
- Response logp：`log_softmax + gather + mask + sum`。
- DPO：policy/ref chosen-rejected log-ratio + `-logsigmoid`。
- REINFORCE：`-adv.detach()*logp`。
- PPO：ratio、两个 surrogate、min、masked mean。
- GRPO：同 prompt reward 标准化 + PPO-style loss。
- Sampling：temperature、top-k、top-p、multinomial。
- GAE：逆序累计 TD residual。

---

# 三周复习执行版

## 第 1 周：基础、Transformer、预训练、SFT

### Day 1

- 复述 MLE—CE—next-token 的统一关系。
- 手写 CE、RMSNorm、训练循环。
- 排查一份故意制造 mask/NaN 错误的代码。

### Day 2～3

- 白板画 Transformer shape、复杂度、RoPE。
- 手写 MHA + causal mask。
- 计算一个具体模型的 KV Cache，并比较 MHA/GQA。

### Day 4

- 讲 BPE、template、预训练/continued PT/SFT。
- 对一个真实数据集做长度、重复、截断统计。

### Day 5～6

- 跑/复盘 SFT、LoRA/QLoRA。
- 手写 assistant mask、LoRA、SFT loss。
- 准备 SFT loss 下降但任务变差的诊断答案。

### Day 7

- 60 分钟串讲 1～4 阶段。
- 进行第一次模拟：Transformer 20 分钟、SFT 20 分钟、代码 20 分钟。

## 第 2 周：RL、RM、PPO、DPO、GRPO、评测

### Day 8

- 推导 Policy Gradient 与 baseline 无偏性。
- 手写 REINFORCE、return、GAE。

### Day 9

- 设计一套 pairwise 标注与 RM 评测。
- 准备 reward hacking 三个具体例子。

### Day 10～11

- 画 PPO 五模型数据流。
- 分析 A 正负时 clipping。
- 手写 PPO loss，读懂 reward/KL/entropy 曲线。

### Day 12

- 从 KL-regularized RL 推到 DPO。
- 手写 response logp 与 DPO loss。
- 对比 sum/mean logp 与 length bias。

### Day 13

- 推导 GRPO group advantage。
- 设计一个可验证 reward，列出 parser 攻击方式。
- 手写 GRPO loss。

### Day 14

- 完成评测 rubric、bootstrap、failure taxonomy。
- 第二次模拟：PPO/DPO/GRPO 对比与曲线诊断。

## 第 3 周：系统、多模态、项目深挖

### Day 15～16

- 拆解显存并手算 Full FT/LoRA。
- 画 ZeRO-1/2/3 与 collective。
- 讲 prefill/decode、PagedAttention、RL 训推分离。

### Day 17

- 画 ViT→Projector/Q-Former→LLM。
- 投 VLA 时再画 action representation 与闭环。

### Day 18～19

- 写 30 秒和 2 分钟项目稿。
- 准备主结果、消融、三个失败、一次 OOM/NaN/吞吐问题。
- 对每个数字记录数据集、样本量、seed 和指标定义。

### Day 20

- 综合模拟：项目 30 分钟、八股 40 分钟、代码 40 分钟。
- 把模糊词全部替换成公式、配置或数字。

### Day 21

- 只补 P0 漏洞。
- 制作一页速查：Attention、LoRA、PG、PPO、DPO、GRPO、显存、ZeRO。
- 保持投递，不等待所有 P1/P2 完成。

---

# 投递前自测题

如果下面任一题无法连续回答 2 分钟，就回到对应章节：

1. 为什么 next-token CE 是最大似然？
2. Attention 为什么除 `sqrt(d)`，Q/K/V 的 shape 是什么？
3. RoPE 为什么能表达相对位置？
4. GQA 如何影响 KV Cache？
5. SFT 为什么通常只监督 assistant tokens？
6. QLoRA 中哪些参数量化、哪些参数训练？
7. Policy Gradient 如何推导，baseline 为什么无偏？
8. RM 的 Bradley–Terry loss 有什么不可识别性？
9. PPO 中 old policy 与 reference policy 有何区别？
10. `A>0` 与 `A<0` 时 PPO clip 分别限制什么？
11. DPO 如何由 KL-regularized RL 推导？
12. DPO 的 sequence logp 为什么有长度偏差？
13. GRPO 为什么不要 value model，代价是什么？
14. 组内 reward 全相同时发生什么？
15. Reward 上升但真实能力下降如何定位？
16. 如何设计不被轻易 hack 的可验证 reward？
17. ZeRO-1/2/3 分别切什么？
18. FlashAttention 降低了什么复杂度，没降低什么？
19. Prefill 与 decode 的瓶颈为什么不同？
20. 同步/异步 RL 的核心 trade-off 是什么？
21. Q-Former 的 learnable queries 训练和推理时怎样变化？
22. 多模态模型如何证明自己真的使用了图像？
23. Open-loop 指标为什么不能替代闭环驾驶？
24. 你的项目最重要消融如何验证核心主张？
25. 你的最好指标是否有方差、置信区间和失败案例？

---

# 核心参考资料

以下优先阅读原论文的摘要、方法和实验/局限，不建议三周内逐页精读全部附录。

## Transformer 与训练

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## Alignment 与 RL

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)
- [Back to Basics: Revisiting REINFORCE Style Optimization for RLHF](https://arxiv.org/abs/2402.14740)

## 系统与多模态

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [An Image is Worth 16x16 Words（ViT）](https://arxiv.org/abs/2010.11929)
- [Learning Transferable Visual Models From Natural Language Supervision（CLIP）](https://arxiv.org/abs/2103.00020)
- [BLIP-2: Q-Former with Frozen Image Encoders and LLMs](https://arxiv.org/abs/2301.12597)

---

# 最后的一条主线

面试时遇到任何后训练问题，都可以沿下面的链条组织：

> **目标是什么 → 数据如何表达目标 → loss/reward 如何把信号传给参数 → reference/baseline 如何控制方差与漂移 → rollout/分布是否匹配 → 系统如何高效执行 → 评测如何防止代理指标欺骗。**

例如 GRPO 不是孤立公式：任务先给出可验证 reward；同 prompt 多采样建立相对 baseline；policy gradient 把序列奖励分配到 token log-prob；clipping 和 reference KL 控制更新；推理引擎承担 rollout；组内全同、staleness、长度偏差会让有效学习信号消失；最终必须用独立正确率、人工/闭环指标验证，而不能只看训练 reward。

真正的“融会贯通”就是：你不仅能说每个模块是什么，还能指出上一个模块的误差怎样传到下一个模块，以及用什么实验把它定位出来。

---

## 原 Checklist 覆盖索引

| 原清单阶段 | 本讲义位置 | 主要产出 |
|---|---|---|
| 1. 数学、DL、PyTorch | 第一阶段 | 公式、排障、CE/RMSNorm/训练循环代码 |
| 2. Transformer | 第二阶段 | shape、RoPE、KV Cache、MoE、Attention 代码 |
| 3. Tokenizer/预训练 | 第三阶段 | BPE、PPL、scaling、数据治理 |
| 4. SFT/PEFT | 第四阶段 | mask、packing、LoRA/QLoRA、SFT 代码 |
| 5. RL 基础 | 第五阶段 | PG 推导、baseline、GAE、LLM 映射 |
| 6. 偏好/RM | 第六阶段 | Bradley–Terry、偏差、hacking、评测 |
| 7. PPO/RLHF | 第七阶段 | 五模型、clip/KL、曲线诊断、代码 |
| 8. DPO | 第八阶段 | 完整推导、长度/噪声、变体、代码 |
| 9. GRPO | 第九阶段 | group advantage、奖励、rollout、异步与代码 |
| 10. 数据/评测 | 第十阶段 | 数据闭环、judge、统计、实验设计 |
| 11. 训练系统 | 第十一阶段 | 显存、ZeRO、推理、PagedAttention、RL infra |
| 12. VLM/VLA | 第十二阶段 | ViT/CLIP/Q-Former、幻觉、动作与闭环 |
| 13. 项目追问 | 第十三阶段 | 30 秒/2 分钟模板、数据/方法/实验/工程问法 |
| 14. 对比题 | 第十四阶段 | 高频对比速答表 |
| 15. 手撕代码 | 第十五阶段 | 全部代码索引、KV Cache、对话 collator |

