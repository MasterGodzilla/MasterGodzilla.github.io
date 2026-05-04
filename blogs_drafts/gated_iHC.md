# mHC 可能不需要 “m”：从 iHC 到 norm-gated iHC 的一个表达能力分析

**作者：GPT-5.5 Pro**

最近一篇知乎文章《你的 deepseek mHC 可能不需要 "m"》提出了一个很有意思的实验观察：在作者报告的 Qwen3 1.7B 和 8B dense from-scratch 实验中，直接把 mHC 的 residual mixer 设成 identity，也就是令 $H^{res}=I$，反而取得了优于 mHC / mHC-lite / orthogonal mHC 的结果 [1]。

这个现象值得认真对待。mHC 的工程实现本来就不算漂亮：它需要动态生成一个 residual mixing matrix，再通过 Sinkhorn-Knopp normalization 近似投影到 Birkhoff polytope。mHC-lite 也指出，有限次 Sinkhorn-Knopp 迭代并不保证精确 doubly stochastic，而且高效实现往往需要额外 CUDA kernel 或专门优化 [3][4]。

所以问题很自然：

**iHC 为什么会 work？它到底损失了 mHC 的哪些表达能力？**

本文试图给这个现象一个表达能力层面的解释。

结论是：

**full-rank mHC 本质上只是 iHC 的 stream-basis gauge。**

也就是说，只要 mHC 的 residual map 不降秩，它并不会带来新的端到端表达能力。

但是：

**rank-deficient mHC 可以做 pure iHC 不能精确完成的 reset-and-reuse。**

所以，pure iHC 虽然很 clean，但它确实少了一个东西：forget / reset。

最终得到的折中方案是 gated iHC。更准确地说，是 norm-gated iHC：

$$
Y_{l+1}
=
D(g_l)Y_l
+
H^{post}_lF_l(H^{pre}_lY_l)
$$

其中 $D(g_l)$ 是一个带 scale normalization 的 diagonal gate。

它保留了 identity HC 的简单性，避免了完整 $n\times n$ residual mixer 和 Sinkhorn-Knopp normalization，同时恢复了 mHC 中真正不可由 full-rank gauge 吸收的部分：让某些 stream direction 被 reset，然后被后续写入复用。



# 1. Preliminaries：mHC 和 iHC 到底是什么？

标准 Transformer residual connection 可以写成：

$$
x_{l+1}
=
x_l
+
F_l(x_l)
$$

Hyper-Connections 的想法是把一条 residual stream 扩展成多条 residual streams [2]。

设第 $l$ 层的 residual state 是 $X_l\in\mathbb R^{n\times d}$，其中 $n$ 是 stream 数量，$d$ 是每条 stream 的 hidden dimension。

一层 mHC 可以抽象写成：

$$
X_{l+1}
=
A_lX_l
+
r_lF_l(p_lX_l)
$$

其中 $A_l=H^{res}_l$ 是 residual stream mixer，$p_l=H^{pre}_l$ 负责从多条 stream 中读出一条输入给 transformer block，$r_l=(H^{post}_l)^T$ 负责把 transformer block 的输出写回多条 stream。

mHC 的核心约束是 $A_l$ 属于 Birkhoff polytope，也就是：

$$
A_l\mathbf 1=\mathbf 1
$$

$$
\mathbf 1^TA_l=\mathbf 1^T
$$

$$
(A_l)_{ij}\ge 0
$$

也就是非负、行和为 1、列和为 1 的 doubly stochastic matrix [3]。

mHC 这么做的动机是恢复 residual connection 里的 identity mapping / norm-preserving 性质，避免原始 HC 中自由 residual mixing 带来的训练不稳定问题 [3]。

但实现上，它通常要通过 Sinkhorn-Knopp normalization 来近似得到 doubly stochastic matrix。mHC-lite 的主要批评正是：有限次 Sinkhorn-Knopp 迭代有 approximation gap，而且高效实现有额外工程门槛 [4]。

iHC 则是最简单的版本：固定 $A_l=I$。于是：

$$
X_{l+1}
=
X_l
+
r_lF_l(p_lX_l)
$$

也就是 residual stream 本身不再跨流混合，每条 stream 保持自己的 identity path。

注意，iHC 并不等于“没有跨流通信”。

因为 $p_l$ 和 $r_l$ 仍然可以由所有 streams flatten 后动态生成。也就是说，跨流信息融合仍然可以发生在 $H^{pre}_l$ 和 $H^{post}_l$ 里。

知乎文章 [1] 的一个重要直觉也是这个：$H^{res}=I$ 只是取消了 residual path 上的显式 stream mixing，并没有取消动态 read/write。



# 2. Theorem 1：full-rank mHC 等价于 iHC

先看最重要的正结果。

## 定理

假设每一层的 mHC residual map 都是静态的、full-rank 的，即 $\det(A_l)\ne 0$。则任意这样的 mHC 网络，都存在一个同深度、同宽度的 iHC 网络，使得二者端到端等价。

更准确地说，full-rank mHC 的 residual mixing 只是 stream basis 的变化。iHC 可以固定 residual path 为 identity，然后把这种变化吸收到新的 read/write map 和输出头里。

## 证明

mHC 的更新是：

$$
X_{l+1}
=
A_lX_l
+
r_lF_l(p_lX_l)
$$

定义累计 stream basis：

$$
P_0=I
$$

$$
P_{l+1}=A_lP_l
$$

因为每个 $A_l$ 都 full-rank，所以每个 $P_l$ 都可逆。

现在令 iHC 的隐藏状态 $Y_l$ 表示同一个 mHC 状态 $X_l$ 在另一个 basis 下的坐标：

$$
X_l=P_lY_l
$$

构造 iHC：

$$
Y_{l+1}
=
Y_l
+
\tilde r_lF_l(\tilde p_lY_l)
$$

令 $\tilde p_l=p_lP_l$，并令 $\tilde r_l=P_{l+1}^{-1}r_l$。

于是 iHC block 看到的输入是：

$$
\tilde p_lY_l
=
p_lP_lY_l
=
p_lX_l
$$

所以它和 mHC 喂给 $F_l$ 的向量完全一样：

$$
F_l(\tilde p_lY_l)
=
F_l(p_lX_l)
$$

再看状态更新：

$$
Y_{l+1}
=
Y_l
+
P_{l+1}^{-1}r_lF_l(p_lX_l)
$$

由于 $X_l=P_lY_l$，所以 $Y_l=P_l^{-1}X_l$。因此：

$$
Y_{l+1}
=
P_l^{-1}X_l
+
P_{l+1}^{-1}r_lF_l(p_lX_l)
$$

又因为 $P_{l+1}=A_lP_l$，所以：

$$
P_l^{-1}
=
P_{l+1}^{-1}A_l
$$

代入可得：

$$
Y_{l+1}
=
P_{l+1}^{-1}A_lX_l
+
P_{l+1}^{-1}r_lF_l(p_lX_l)
$$

也就是：

$$
Y_{l+1}
=
P_{l+1}^{-1}
\left(
A_lX_l
+
r_lF_l(p_lX_l)
\right)
$$

所以：

$$
Y_{l+1}
=
P_{l+1}^{-1}X_{l+1}
$$

等价于：

$$
X_{l+1}
=
P_{l+1}Y_{l+1}
$$

归纳得到，$X_l=P_lY_l$ 对每一层都成立。

如果最后输出是线性读出 $o=CX_L$，那么 iHC 用 $\tilde C=CP_L$ 即可得到：

$$
\tilde CY_L
=
CP_LY_L
=
CX_L
$$

证毕。

## 解释

这个 theorem 说明：full-rank mHC 没有真正超过 iHC。

full-rank residual mixer 做的是 change of basis，不是 new computation。

它把 stream 坐标系转来转去。但只要这个坐标变换可逆，iHC 就可以换一套 read/write 参数，把同样的 computation 写出来。

这也解释了为什么简单的 permutation 不能构成反例。

如果 $A_l$ 只是一个 permutation matrix，那更明显。它只是把 stream 编号换了。embedding、read、write、输出头都可以吸收这个重排。

所以 mHC 中真正可能超过 iHC 的部分，不是 full-rank mixing。

而是 rank collapse。



# 3. mHC 真正多出来的能力：reset and reuse

full-rank matrix 是换 basis。

rank-deficient matrix 是丢信息。

如果 mHC 允许 $rank(A_l)<n$，它就可以做一件 pure iHC 很难做的事：

**erase old stream direction, then reuse it。**

这不是 gauge。

这是 reset。

知乎文章 [1] 中提到的 doubly stochastic matrix 累积坍缩，也可以从这个角度理解：如果 residual mixing 在多层后逐渐压到低维方向，那么它确实带来了某种 forgetting / homogenization。但这种 forgetting 是不是我们想要的，以及是不是需要完整 Sinkhorn matrix 来实现，就是另一个问题。

下面给一个小反例，说明 rank-deficient mHC 的 reset-and-reuse 能力确实不是 pure iHC 能精确吸收的。



# 4. Counterexample：rank-deficient mHC 可以，pure iHC 不可以

为了让反例和 transformer FFN 更接近，我们不用简单的平方函数，而用一个 scalar no-bias SwiGLU primitive：

$$
\phi(t)
=
t\cdot swish(t)
=
t^2\sigma(t)
$$

SwiGLU 属于 GLU variants；Shazeer 的 GLU variants paper 讨论了把 GLU / GEGLU / SwiGLU 等形式用于 Transformer FFN，并报告了相对 ReLU / GELU 的改进 [6]。

这个反例不是说真实大模型里的宽 MLP 不能近似某个函数。它说明，在相同 depth、相同 residual-stream 机制、相同 scalar no-bias SwiGLU primitive 下，rank-deficient residual mixer 提供了一种 pure identity residual path 没有的精确操作。

设 $n=2$，$d=1$。

定义 $e=(1,1)^T$，$q=(1,-1)^T$，以及 rank-1 averaging matrix：

$$
J=\frac12ee^T
$$

它是 doubly stochastic，但不是 full-rank：

$$
rank(J)=1
$$

并且：

$$
Jq=0
$$

也就是说，它会杀掉 zero-sum direction。

令输入为 $x=(x_1,x_2)^T$，定义差分坐标 $Q=q^Tx=x_1-x_2$。

## mHC 构造

第一层：

$$
X_1
=
Jx
+
q\phi(q^Tx)
$$

也就是先用 $J$ 擦掉 raw difference direction，再把新的 feature 写进 $q$ direction。

第二层：

$$
X_2
=
X_1
+
\frac12e
\phi
\left(
\frac12q^TX_1
\right)
$$

现在计算第二层读到的东西：

$$
\frac12q^TX_1
=
\frac12q^TJx
+
\frac12q^Tq\phi(Q)
$$

因为 $q^TJx=0$，且 $q^Tq=2$，所以：

$$
\frac12q^TX_1
=
\phi(Q)
$$

最后用 stream-sum readout：

$$
o_{mHC}(x)
=
e^TX_2
$$

得到：

$$
o_{mHC}(x)
=
e^Tx
+
\phi(\phi(q^Tx))
$$

也就是：

$$
o_{mHC}(x)
=
x_1+x_2+\phi(\phi(x_1-x_2))
$$

所以 rank-deficient mHC 精确计算了：

$$
o_{mHC}(x)
=
x_1+x_2+\phi(\phi(x_1-x_2))
$$

它的机制是：

$$
Q
\rightarrow
\text{erase raw }Q
\rightarrow
\text{store }\phi(Q)
\rightarrow
\phi(\phi(Q))
$$

## pure iHC 的一般形式

同样两层 pure iHC 写成：

$$
Y_1
=
x
+
uG_1(a^Tx)
$$

$$
Y_2
=
Y_1
+
vG_2(b^TY_1)
$$

其中 $G_1$ 和 $G_2$ 是任意有限宽 no-bias scalar SwiGLU block。

输出头允许任意线性 head：

$$
o_{iHC}(x)
=
h^TY_2
$$

因为 no-bias SwiGLU 在原点附近没有一次项：

$$
G(z)=O(z^2)
$$

目标函数的一次项是 $x_1+x_2=e^Tx$，所以输出 head 的线性部分被迫满足 $h=e$。

于是 iHC 的非线性部分一定形如：

$$
N_{iHC}(x)
=
\alpha G_1(A)
+
\beta G_2(B+\gamma G_1(A))
$$

其中 $A=a^Tx$，$B=b^Tx$，$\alpha=e^Tu$，$\beta=e^Tv$，$\gamma=b^Tu$。

我们要求：

$$
N_{iHC}(x)
=
\phi(\phi(q^Tx))
$$

对所有 $x$ 成立。

## 沿着 zero-mean 方向看

取：

$$
x=
\frac Q2q
=
(Q/2,-Q/2)^T
$$

于是 $q^Tx=Q$，且 $e^Tx=0$。

沿着这条线，$A$ 和 $B$ 都是 $Q$ 的线性函数：

$$
A=cQ
$$

$$
B=dQ
$$

因此必须有：

$$
\alpha G_1(cQ)
+
\beta G_2(dQ+\gamma G_1(cQ))
=
\phi(\phi(Q))
$$

对所有 $Q$ 成立。

现在看 $Q\to+\infty$。

SwiGLU primitive 满足：

$$
\phi(Q)
=
Q^2\sigma(Q)
\sim Q^2
$$

因此：

$$
\phi(\phi(Q))
\sim Q^4
$$

更关键的是，它的 positive-tail 没有 $Q^3$ polynomial term。

而 iHC 要产生 $Q^4$，必须让第一层产生一个二次尾项：

$$
G_1(cQ)
\sim
\kappa Q^2
$$

其中 $\kappa\ne 0$。

第二层读到的是：

$$
dQ+\gamma G_1(cQ)
\sim
dQ+\gamma\kappa Q^2
$$

第二个 SwiGLU block 的正向尾部也是二次，所以：

$$
G_2(dQ+\gamma G_1(cQ))
\sim
\lambda(dQ+\gamma\kappa Q^2)^2
$$

展开：

$$
\lambda(dQ+\gamma\kappa Q^2)^2
=
\lambda\gamma^2\kappa^2Q^4
+
2\lambda d\gamma\kappa Q^3
+
\lambda d^2Q^2
$$

要匹配目标的 $Q^4$，必须有 $\beta\lambda\gamma\kappa\ne 0$。

但目标没有 $Q^3$ 项，所以必须有 $d=0$。

也就是说，第二层 read 不能读 raw $Q$ direction。

在二维里，$d=0$ 等价于 $b^Tq=0$，所以 $b$ 平行于 $e$。

另一方面，目标也没有 $Q^2$ tail，所以第一层直接对输出可见的那部分必须消失：

$$
\alpha=e^Tu=0
$$

在二维里，这意味着 $u$ 平行于 $q$。

于是：

$$
\gamma=b^Tu=0
$$

但前面为了产生 $Q^4$，我们又必须有 $\gamma\ne 0$。

矛盾。

所以，这个 rank-deficient mHC 函数无法由同深度 pure iHC 精确参数化。



# 5. 反例的直觉

mHC 做的是：

$$
\text{old }Q
\rightarrow
0
\rightarrow
\phi(Q)
$$

所以第二层读到的是干净的 $\phi(Q)$。

pure iHC 做不到 erase。

它只能保留：

$$
Q+\text{new feature}
$$

所以第二层读到的一般是：

$$
dQ+\gamma\phi(Q)
$$

再过一个 SwiGLU，就会产生污染项：

$$
(dQ+\gamma Q^2)^2
=
\gamma^2Q^4
+
2d\gamma Q^3
+
d^2Q^2
$$

这个 $Q^3$ 就是 raw residual stream 没有被 reset 的代价。

所以 mHC 真正多出来的不是 full-rank mixing，而是 finite reset and reuse。



# 6. 先得到 gated iHC：iHC + per-direction reset

既然 pure iHC 缺的是 reset，一个直接的修正就是 gated iHC：

$$
Y_{l+1}
=
D_lY_l
+
\tilde r_lF_l(\tilde p_lY_l)
$$

其中 $D_l$ 是 diagonal gate：

$$
D_l
=
diag(g_{l,1},g_{l,2},...,g_{l,n})
$$

直觉上，$g_{l,i}=1$ 表示这个 stream direction 继续活着；$g_{l,i}=0$ 表示这个 stream direction 被 reset，可以复用。

表达能力分析里，核心不是 gate 的具体连续数值，而是 zero pattern：

$$
g_{l,i}=0
\quad\text{or}\quad
g_{l,i}\ne0
$$

也就是 survive / die。

这就是 gated iHC 的核心：

$$
\text{iHC}
+
\text{per-direction reset}
$$

## Theorem 2：diagonal-gated iHC 可以表示任意 static mHC

给定任意 static mHC：

$$
X_{l+1}
=
A_lX_l
+
r_lF_l(p_lX_l)
$$

其中每个 $A_l$ 是任意线性 residual mixer。特别地，它可以是 mHC 的 doubly stochastic matrix。

则存在一个同深度 diagonal-gated iHC：

$$
Y_{l+1}
=
D_lY_l
+
\tilde r_lF_l(\tilde p_lY_l)
$$

以及每一层的 stream basis 变换 $P_l$，使得 $X_l=P_lY_l$，并且端到端输出等价。

### 关键线性代数 lemma

任意有限线性链：

$$
V_0
\to
V_1
\to
V_2
\to
\cdots
\to
V_L
$$

其中相邻空间之间的线性映射分别是 $A_0,A_1,...,A_{L-1}$。那么可以选取每个 $V_l$ 上的 basis，使得每个 $A_l$ 在这些 basis 下都是 diagonal 0/1 gate。

也就是说，存在可逆矩阵 $P_0,P_1,...,P_L$，以及 diagonal matrix $E_0,E_1,...,E_{L-1}$，使得：

$$
A_lP_l=P_{l+1}E_l
$$

其中：

$$
E_l=diag(0,1,1,0,...)
$$

这其实就是有限线性链的 interval decomposition / barcode normal form。

每个 basis vector 对应一个 interval：它在某一层出生，经过若干层，然后在某一层死亡。

在这个 basis 下，每个 residual map 对一个 basis direction 只做两件事之一：要么把它送到下一层同一个 interval 的 basis direction，要么 kill it。

这正是 diagonal gate。

### 用 lemma 证明模拟

由 lemma，存在：

$$
A_lP_l=P_{l+1}E_l
$$

设 $X_l=P_lY_l$。

构造 gated iHC 的 read/write：

$$
\tilde p_l=p_lP_l
$$

$$
\tilde r_l=P_{l+1}^{-1}r_l
$$

那么 block 输入一致：

$$
\tilde p_lY_l
=
p_lP_lY_l
=
p_lX_l
$$

所以：

$$
F_l(\tilde p_lY_l)
=
F_l(p_lX_l)
$$

gated iHC 更新后乘回 $P_{l+1}$：

$$
P_{l+1}Y_{l+1}
=
P_{l+1}E_lY_l
+
P_{l+1}\tilde r_lF_l(\tilde p_lY_l)
$$

利用 $A_lP_l=P_{l+1}E_l$ 和 $\tilde r_l=P_{l+1}^{-1}r_l$，得到：

$$
P_{l+1}Y_{l+1}
=
A_lP_lY_l
+
r_lF_l(p_lX_l)
$$

也就是：

$$
P_{l+1}Y_{l+1}
=
A_lX_l
+
r_lF_l(p_lX_l)
=
X_{l+1}
$$

所以归纳得到，$X_l=P_lY_l$ 对每层成立。

如果 mHC 最后的输出是 $o=CX_L$，那么 gated iHC 用 $\tilde C=CP_L$ 即可：

$$
\tilde CY_L
=
CP_LY_L
=
CX_L
$$

证毕。

## 这个 theorem 到底说了什么？

mHC residual matrix $A_l$ 看起来很 general：它是一个 $n\times n$ matrix。

但从跨层线性链的角度，它的本质只有两部分。

第一，full-rank part：change basis。

第二，rank-deficient part：kill directions。

第一部分 iHC 已经能吸收。

第二部分需要 gate。

所以可以把 mHC 理解成：

$$
\text{mHC}
=
\text{iHC}
+
\text{per-direction reset}
$$

这就是 gated iHC 的意义。

它不是把 mHC 的 Sinkhorn machinery 原样搬过来。

它直接保留 mHC 真正有表达力意义的东西：哪些 stream directions 应该 survive，哪些 stream directions 应该 reset and reuse。



# 7. gated iHC 的训练问题：gate 会导致 gradient vanishing

上面的 gated iHC 解决了表达力问题，但还没有完全解决训练稳定性问题。

如果直接使用：

$$
Y_{l+1}
=
g_l\odot Y_l
+
\text{write}
$$

那么 residual path 的 Jacobian 里会不断出现 diagonal gate。

跨很多层之后，大致会出现：

$$
D_{L-1}D_{L-2}\cdots D_0
$$

如果很多 $g_{l,i}<1$，那么 surviving path 的梯度也可能被不断缩小。

这和普通 RNN 里反复乘 transition matrix 导致 gradient vanishing 是同一种问题。mHC 之所以强调 Birkhoff / norm-preserving / identity-like residual path，本质上也是为了避免 residual path 在深层里乱掉 [3]。

所以 gated iHC 还需要一个 scale-stability 设计。

有两个自然方向：

1. mean-preserving gate

2. norm-preserving gate

下面分别看。



# 8. mean-preserving gate：对齐 mHC，但不是最好的答案

如果想模仿 mHC 的 doubly stochastic 结构，一个自然设计是 mean-preserving gate。

设：

$$
J=\frac1n\mathbf 1\mathbf 1^T
$$

它把所有 streams 投影到 mean direction。

一个最简单的 mean-preserving gate 是：

$$
A(g)=J+g(I-J)
$$

它的含义是：

$$
Y=\text{mean part}+\text{difference part}
$$

mean part 不动，difference part 乘 $g$。

这个设计有两个好处。

第一，它保留 stream mean：

$$
\mathbf 1^TA(g)=\mathbf 1^T
$$

第二，当 $0\le g\le 1$ 时，它不会放大 zero-sum difference modes。

所以它很像 mHC 的 Birkhoff intuition：保留均值方向，收缩差分方向。

但这个设计不适合作为最终答案。

原因一：scalar mean-preserving gate 太弱。

它只能统一缩放所有 zero-sum directions。比如 $n=3$ 时，它不能擦掉一个 zero-sum direction，同时保留另一个 zero-sum direction。

而 mHC 可以做到这种 partial reset。

原因二：vector mean-preserving gate 会重新引入混合。

如果希望不同 difference directions 有不同 gate，同时还严格保留 mean，通常需要先把 hidden state 分解到某个 zero-mean basis，再 gate，再投回原 stream basis。

这已经不再是简单的 per-stream diagonal gate，而是重新引入了 basis transform / projection / mixing。

这和我们想要的 clean iHC 方向相反。

原因三：mean preservation 不等于 gradient scale preservation。

$A(g)=J+g(I-J)$ 确实保留 mean direction，但所有 difference directions 仍然会被 $g$ 连乘衰减。也就是说，它只是让一个 mean anchor 不死，并没有解决所有 useful stream directions 的 gradient vanishing。

所以 mean-preserving gate 更像是在模仿 mHC 的约束，而不是直接解决 gated iHC 的训练问题。

简短地说：

**mean-preserving gate 对齐了 mHC，但太受限，也不是最直接的 scale-stability 目标。**



# 9. norm-preserving gate：更适合 gated iHC

如果我们关心的是训练稳定性，那么更直接的目标不是 preserve mean，而是 preserve residual path 的 scale。

但这里有一个坑。

如果要求一个线性 gate $R$ 对所有 hidden state 都严格保范数：

$$
\|RY\|=\|Y\|
$$

那它必须是 full-rank。

特别地，如果 $R$ 是 diagonal matrix：

$$
R=diag(d_1,d_2,...,d_n)
$$

则：

$$
\|RY\|^2
=
\sum_i d_i^2\|Y_i\|^2
$$

要对所有 $Y$ 都等于 $\sum_i \|Y_i\|^2$，就必须有：

$$
|d_i|=1
$$

这意味着它不能有 $d_i=0$，也就不能 reset。

所以，严格的 state-norm-preserving 和 reset 是冲突的。

我们真正想要的是一种 weaker but useful 的 norm preservation：

**preserve the RMS scale of the gate operator, while still allowing zero entries.**

也就是：

$$
D(g)
=
c(g)\,diag(g)
$$

其中：

$$
c(g)=\frac{\sqrt n}{\|g\|_2}
$$

这样有：

$$
\frac1n\|D(g)\|_F^2=1
$$

如果 $g$ 是 binary mask，并且有 $k$ 个 alive streams，那么：

$$
D(g)
=
\sqrt{\frac nk}\,diag(g)
$$

例如 $n=4$，只保留两个 streams，那么 surviving streams 会乘上 $\sqrt 2$。

这不是对每个具体 hidden state 严格保范数，而是保持 residual gate 的 average RMS scale。

它允许 reset，同时避免 gate 数量变少时 residual path 的整体 scale 系统性下降。

这就是 norm-gated iHC：

$$
Y_{l+1}
=
D(g_l)Y_l
+
H^{post}_lF_l(H^{pre}_lY_l)
$$

其中：

$$
D(g_l)=c(g_l)\,diag(g_l)
$$

$$
c(g_l)=\frac{\sqrt n}{\|g_l\|_2}
$$

实际实现里，为了数值稳定可以写成：

$$
c(g_l)=\frac{\sqrt n}{\|g_l\|_2+\epsilon}
$$

也可以加 cap：

$$
c(g_l)=\min\left(c_{max},\frac{\sqrt n}{\|g_l\|_2+\epsilon}\right)
$$

这样避免当 alive streams 太少时 surviving streams 被过度放大。



# 10. Theorem 3：norm-gated iHC 和普通 gated iHC 表达力等价

现在证明：norm normalization 不改变表达力。

普通 diagonal-gated iHC 使用的是 binary mask：

$$
E_l=diag(m_l)
$$

其中 $m_l\in\{0,1\}^n$。

norm-gated iHC 使用的是：

$$
D_l=c_lE_l
$$

其中 $c_l$ 是非零标量。例如：

$$
c_l=\sqrt{\frac n{k_l}}
$$

这里 $k_l$ 是 alive stream 数量。

关键点是：

**非零标量可以被跨层 basis scaling 吸收。**

假设普通 gated chain 里有：

$$
Z_{l+1}
=
E_lZ_l+\text{write}
$$

现在用 norm-gated chain：

$$
Y_{l+1}
=
D_lY_l+\text{write}
$$

来模拟它。

令：

$$
Z_l=s_lY_l
$$

我们希望 residual path 匹配：

$$
s_{l+1}D_lY_l=E_lZ_l
$$

代入 $Z_l=s_lY_l$ 和 $D_l=c_lE_l$，得到：

$$
s_{l+1}c_lE_lY_l=s_lE_lY_l
$$

只要令：

$$
s_{l+1}=\frac{s_l}{c_l}
$$

就成立。

所以每一层的 norm-normalization scalar 都可以被 hidden basis 的层间缩放吃掉。

因此：

$$
\text{ordinary diagonal-gated iHC}
=
\text{norm-gated iHC}
$$

表达力相同。

现在结合 Theorem 2：ordinary diagonal-gated iHC 可以表示任意 static mHC。

于是：

$$
\text{static mHC}
\subseteq
\text{norm-gated iHC}
$$

更直接地，也可以把 Theorem 2 的 normal form 改写成 norm-gated 版本。

由 normal form，存在 $P_l$ 和 $E_l$，使得：

$$
A_lP_l=P_{l+1}E_l
$$

令：

$$
D_l=c_lE_l
$$

并定义新的 basis：

$$
Q_l=s_lP_l
$$

其中：

$$
s_0=1
$$

$$
s_{l+1}=\frac{s_l}{c_l}
$$

那么：

$$
A_lQ_l=Q_{l+1}D_l
$$

这就把普通 diagonal gate normal form 变成了 norm-gated normal form。

接着令：

$$
X_l=Q_lY_l
$$

并构造：

$$
\tilde p_l=p_lQ_l
$$

$$
\tilde r_l=Q_{l+1}^{-1}r_l
$$

即可得到和 Theorem 2 完全一样的模拟证明。

所以 norm-gated iHC 既保留了 gated iHC 的表达能力，又更适合训练稳定性。



# 11. mean vs norm：为什么最后选 norm-gated iHC？

现在可以把三个版本放在一起看。

pure iHC：

$$
Y_{l+1}
=
Y_l+\text{write}
$$

优点是最 clean，identity path 完全稳定。

缺点是没有 reset，不能精确实现 rank-deficient mHC 的 reset-and-reuse。

ordinary gated iHC：

$$
Y_{l+1}
=
g_l\odot Y_l+\text{write}
$$

优点是有 reset，表达力上足够覆盖 static mHC。

缺点是如果很多 gate 小于 1，residual path 的 scale 和梯度可能会衰减。

mean-preserving gated iHC：

$$
Y_{l+1}
=
\left(J+g_l(I-J)\right)Y_l+\text{write}
$$

优点是贴近 mHC 的 doubly-stochastic / mean-preserving intuition。

缺点是 scalar 版本太弱；vector 版本又会重新引入 projection / mixing；而且它保留的是 mean，不是所有 useful directions 的 scale。

norm-gated iHC：

$$
Y_{l+1}
=
D(g_l)Y_l+\text{write}
$$

其中：

$$
D(g_l)=c(g_l)\,diag(g_l)
$$

$$
c(g_l)=\frac{\sqrt n}{\|g_l\|_2+\epsilon}
$$

优点是保留 reset，同时保持 residual gate 的 RMS scale。

更重要的是，它和 ordinary gated iHC 表达力等价，因为 normalization scalar 可以被 basis scaling 吸收。

所以最终推荐的是：

**先用 gated iHC 恢复 reset-and-reuse，再用 norm normalization 稳定 residual path scale。**

一句话：

**mean-preserving 是为了模仿 mHC；norm-gating 是为了让 gated iHC 自己训练稳定。**

所以我更倾向于 norm-gated iHC。



# 12. dynamic mHC 的 caveat

上面的 theorem 是 static residual maps 的严格版本。

真实 mHC 里，$A_l$ 可以由 hidden state 动态生成；mHC 论文中 $H^{pre}$、$H^{post}$、$H^{res}$ 都可以来自动态 projection，$H^{res}$ 再经过 Sinkhorn-Knopp 得到受约束的 residual map [3]。

动态情况下，basis $P_l$ 也会变成输入相关：

$$
P_l=P_l(x)
$$

这样 exact gauge proof 就不再是一个固定线性 reparameterization。

但是这反而强化了 norm-gated iHC 的设计方向：

不要让 controller 生成一个完整 $n\times n$ matrix，然后 Sinkhorn。

让 controller 直接生成一个 per-stream gate：

$$
g_l(x)\in[0,1]^n
$$

再加上动态 read/write：$\tilde p_l(x)$ 和 $\tilde r_l(x)$。

然后用 norm normalization 保持 residual gate 的 RMS scale。

这比动态 Birkhoff projection 更简单。



# 13. 实践建议：norm-gated iHC 怎么做？

我会从这个版本开始：

$$
Y_{l+1}
=
D(g_l)Y_l
+
H^{post}_lF_l(H^{pre}_lY_l)
$$

其中：

$$
D(g_l)=c(g_l)\,diag(g_l)
$$

$$
c(g_l)=\frac{\sqrt n}{\|g_l\|_2+\epsilon}
$$

如果担心 surviving streams 被过度放大，可以用 capped version：

$$
c(g_l)
=
\min\left(c_{max},\frac{\sqrt n}{\|g_l\|_2+\epsilon}\right)
$$

初始化时让：

$$
g_l\approx \mathbf 1
$$

这样模型一开始接近 pure iHC。

训练过程中，如果模型发现某个 stream direction 需要 reset，就把对应 gate 降下来。

为了精确表示 rank collapse，gate 最好能真的到 0。普通 sigmoid 只能给 $g_{l,i}\in(0,1)$，所以它只能近似 reset，不能精确 reset。

可以考虑 hard-sigmoid、clipped gate、straight-through binary gate，或者 $\ell_0$-style gate。

如果担心所有 stream 都被杀掉，可以保留一个 always-on anchor direction：

$$
g_{l,0}=1
$$

然后只 gate 其他 streams：

$$
g_l=(1,g_{l,2},...,g_{l,n})
$$

这个 anchor direction 不一定要承担主要计算。它更像一个稳定的 residual backbone。



# 14. 为什么这个结论很自然：RNN 到 LSTM 的类比

这件事其实很像 RNN 到 LSTM。

普通 RNN 的问题不是“表达不了任意函数”，而是 state update 太不受控：

$$
h_t
=
f(W_hh_{t-1}+W_xx_t)
$$

信息保留、信息覆盖、梯度传播，全混在一个 nonlinear update 里。

LSTM 的核心不是更复杂的 nonlinearity。

核心是把 state update 拆成 forget 和 write。

典型形式是：

$$
c_t
=
f_t\odot c_{t-1}
+
i_t\odot\tilde c_t
$$

Hochreiter 和 Schmidhuber 的 LSTM 原文是为了解决长时间依赖和梯度衰减问题；后来的 forget gate 也成为现代 LSTM 里最关键的 reset mechanism 之一 [7]。

对 residual stream 也是一样。

pure iHC 是：

$$
Y_{l+1}
=
Y_l
+
\text{write}
$$

它只有 write，没有 forget。

mHC 是：

$$
X_{l+1}
=
A_lX_l
+
\text{write}
$$

它有 forget / mix / reset，但用的是一个完整 matrix。

norm-gated iHC 是：

$$
Y_{l+1}
=
D(g_l)Y_l
+
\text{write}
$$

它有 forget，也有 scale normalization。

这就是 residual-stream 版本的 LSTM cell。



# 15. 再看横着的架构：gated linear attention / Gated DeltaNet

同一个故事也发生在线性注意力和 state-space-like 架构里。

linear attention 可以看成一个横向 RNN：

$$
S_t
=
S_{t-1}
+
v_tk_t^T
$$

读出：

$$
o_t
=
S_tq_t
$$

这和 pure iHC 很像：状态永远 identity carry，然后不断 additive write。

问题也类似：如果状态只能加，不能忘，就会污染。

Gated Linear Attention 的方向是加入 data-dependent gates，使线性注意力具有更强的 adaptive memory control，同时保持 linear attention 的并行训练 / 线性推理优势 [8]。

Gated DeltaNet 也明确把 gating 和 delta update 分开理解：gating 负责 adaptive memory erasure，delta rule 负责 targeted update [9]。

这和我们这里的结论完全平行：

$$
\text{pure additive state}
\Rightarrow
\text{clean but cannot reset}
$$

$$
\text{full matrix transition}
\Rightarrow
\text{expressive but complex}
$$

$$
\text{diagonal gate + write + scale normalization}
\Rightarrow
\text{clean and enough}
$$

所以 norm-gated iHC 不是一个孤立 trick。

它是很多横向架构里反复出现的模式：

$$
\text{identity memory path}
+
\text{learned forget gate}
+
\text{scale control}
+
\text{targeted write}
$$



# 16. 最后总结

mHC 有价值。

它告诉我们：多 residual streams 是有用的，动态 read/write 是有用的，stream-level routing 是有用的。

但从表达能力上看，full-rank residual mixing 只是 gauge：

$$
\text{full-rank mHC}
=
\text{iHC under stream-basis change}
$$

真正不能被 pure iHC 吸收的是 rank collapse：

$$
\text{erase old direction and reuse it}
$$

而这件事不需要完整的 $n\times n$ Sinkhorn matrix。

一个 per-stream diagonal gate 就够了：

$$
Y_{l+1}
=
g_l\odot Y_l
+
H^{post}_lF_l(H^{pre}_lY_l)
$$

但是，直接 gate 可能带来 residual path 的 scale decay 和 gradient vanishing。因此更合适的工程版本是 norm-gated iHC：

$$
Y_{l+1}
=
D(g_l)Y_l
+
H^{post}_lF_l(H^{pre}_lY_l)
$$

其中：

$$
D(g_l)=c(g_l)\,diag(g_l)
$$

$$
c(g_l)=\frac{\sqrt n}{\|g_l\|_2+\epsilon}
$$

如果要一句话概括：

**mHC 的本质不是 mixing，而是 forgetting；gated iHC 保留 forgetting，norm-gated iHC 进一步稳定 scale。**

所以结论是：

**identity HC 是一个很强的 baseline；norm-gated iHC 是更自然的下一步。**



# References

[1] 涮月亮的谪仙人. **你的 deepseek mHC 可能不需要 "m"**. 知乎专栏, edited 2026-02-28. https://zhuanlan.zhihu.com/p/2010852389670908320

[2] Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou. **Hyper-Connections**. arXiv:2409.19606. https://arxiv.org/abs/2409.19606

[3] Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang. **mHC: Manifold-Constrained Hyper-Connections**. arXiv:2512.24880. https://arxiv.org/abs/2512.24880

[4] Yongyi Yang, Jianyang Gao. **mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations**. arXiv:2601.05732. https://arxiv.org/abs/2601.05732

[5] Richard Sinkhorn, Paul Knopp. **Concerning Nonnegative Matrices and Doubly Stochastic Matrices**. Pacific Journal of Mathematics, 1967. https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf

[6] Noam Shazeer. **GLU Variants Improve Transformer**. arXiv:2002.05202. https://arxiv.org/abs/2002.05202

[7] Sepp Hochreiter, Jürgen Schmidhuber. **Long Short-Term Memory**. Neural Computation, 1997. https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory

[8] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. **Gated Linear Attention Transformers with Hardware-Efficient Training**. arXiv:2312.06635. https://arxiv.org/abs/2312.06635

[9] Songlin Yang, Jan Kautz, Ali Hatamizadeh. **Gated Delta Networks: Improving Mamba2 with Delta Rule**. arXiv:2412.06464. https://arxiv.org/abs/2412.06464