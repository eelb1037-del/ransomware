# 跨平台勒索软件研究总结:x86 Windows / ARM / Apple

本文从勒索软件的**真实威胁现状**出发,对比三大平台(x86 Windows、ARM、Apple),
并落到本工程**基于 PMU 硬件事件检测**的可迁移性。

> 说明:以下为公开威胁情报与体系结构知识的归纳,用于指导检测方案设计;不含任何
> 攻击实现细节。

## 一、总览对比

| 维度 | x86 Windows | ARM(Linux/嵌入式/Android) | Apple(macOS / Apple Silicon) |
|------|------------|---------------------------|------------------------------|
| 威胁程度 | **极高**(勒索主战场) | 中(上升快,服务器/NAS/IoT) | 低但增长(定向、供应链) |
| 典型目标 | 企业终端、AD 域、文件服务器 | NAS、云原生、虚拟化宿主、路由器 | 开发者、企业 Mac 群、Xcode 链 |
| 代表家族 | LockBit、BlackCat/ALPHV、Conti、Ryuk、WannaCry | RansomEXX(ARM 版)、针对 ESXi/QNAP 的变种 | KeRanger、EvilQuest/ThiefQuest、LockBit macOS 测试版 |
| 加密指令 | AES-NI(x86 专用加密扩展) | ARMv8 Crypto Extension(AES/SHA) | 同 ARMv8 Crypto + Apple 自研加速 |
| 落地难度 | 低(生态成熟、工具链多) | 中(需交叉编译、适配多) | 高(签名/公证/SIP/沙箱阻力大) |

## 二、x86 Windows——勒索软件的主战场

**为什么集中在这里**:装机量最大、企业 AD 域便于横向移动、历史漏洞多、加密货币
变现成熟。占全球勒索事件的绝大多数。

**典型攻击链**:钓鱼/RDP 暴破/漏洞利用 → 提权 → 横向移动(SMB/PsExec) →
**删除卷影副本(VSS)** → 关闭备份/安全服务 → 全盘加密 → 勒索信。

**硬件层可观测指纹(本工程关注点)**:
- **AES-NI 指令暴增**:x86 用 `AESENC`/`AESDEC` 专用指令,对应 PMU 事件可数(如
  Intel 的 FP/AES 相关计数器);
- **全盘遍历 → 各级缓存未命中飙升**(与 ARM 一致的物理规律);
- **写突发**:密文大量写回,store 占比 + 写带宽抬升;
- **熵升高**:加密后文件熵接近随机(需文件层信号,非纯硬件)。

**对应到本工程**:特征思想可迁移,但 **PMU 事件编号是 x86 体系(perf 的 `cpu/event=`
或 Intel PMU),与 ARMv8 不同**——`crypto_spec` 要换成 x86 的 AES 相关计数器。
缓存/分支/指令构成类特征(MPKI、IPC、store 占比)概念通用。

## 三、ARM——增长最快的新战场

**场景**:云原生(ARM 服务器如 AWS Graviton)、虚拟化宿主(ESXi on ARM)、
**NAS/QNAP/群晖**(常用 ARM SoC)、IoT/路由器、Android。NAS 因存大量数据、
常年在线、安全更新滞后,成为重灾区。

**特点**:
- 攻击者把 Windows 家族**交叉编译**出 ARM/Linux 版(如 RansomEXX、针对 ESXi 的变种);
- 目标常是**整卷/整阵列加密**,而非单机文件;
- Android 勒索多为"锁屏勒索"(覆盖屏幕)而非真加密,危害相对低。

**硬件层指纹(本工程的原生战场)**:
- **ARMv8 Crypto Extension**:`AESE`/`AESD`/`SHA` 指令 → 本工程的 `crypto_spec`
  (架构事件 0x77 系列)正是为此设计;
- 全盘遍历 → `ll_cache_miss`/`l1d_cache_refill` 飙升(本工程 12 档核心特征);
- 这些正是 `data_gen.py` 建模的指纹。

**对应到本工程**:**当前实现就是面向 ARMv8-A 的**,迁移成本最低。⚠️ 但需注意:
- 真实 ARM 核心通常仅 6 个通用 PMU 计数器(见 [counters_6.md](counters_6.md)),
  采 13 信号需多路复用;
- 不同 ARM 实现(Neoverse 服务器 / Cortex / 厂商 SoC)事件编号可能有差异,需
  `perf list` 核对。

## 四、Apple(macOS / Apple Silicon)——阻力最大但在增长

**现状**:历史上 Mac 勒索少(KeRanger 2016、EvilQuest/ThiefQuest 2020),但随
Mac 进入企业、开发者价值高,**定向攻击与供应链(Xcode/依赖投毒)风险上升**;
LockBit 已放出 macOS/Apple Silicon 测试样本(成熟度不高)。

**Apple 的多层防御抬高了落地门槛**:
- **Gatekeeper + 公证(notarization)**:未签名/未公证程序默认拦截;
- **SIP(系统完整性保护)**:保护系统目录;
- **TCC**:访问"文档/桌面/下载"等需用户授权,阻碍静默全盘加密;
- **App 沙箱 + 只读系统卷(SSV)**。
因此 Mac 勒索常需社工诱导授权,或走开发者/供应链路径。

**硬件层指纹(Apple Silicon 的特殊性)**:
- Apple Silicon 是 ARMv8/v9,**有 Crypto Extension**,但 **PMU 是 Apple 私有实现**:
  事件编号与语义**不同于 ARM 标准**,且通过 `kperf`/`kpc`(私有框架)或 Instruments
  访问,`crypto_spec` 这类不一定暴露;
- 还有专用 AMX/加速器与硬件 AES 引擎,部分加密可能**不体现在通用指令计数**上
  (走专用引擎),削弱"crypto 指令占比"信号——这是 Apple 平台的检测难点。

**对应到本工程**:特征**思想**可迁移,但
- 不能直接套用 ARMv8 架构事件编号,**必须用 Apple PMU 的实际事件**重新映射;
- `crypto_spec` 信号可能因硬件 AES 引擎而减弱,需更依赖**缓存遍历 + 写突发 + 节奏**
  类特征(恰好是本工程 12 档与差分特征的强项);
- 采集需 `kpc`/Instruments,权限与稳定性是工程难点。

## 五、跨平台共性:为什么 PMU 检测有普适价值

无论 x86 / ARM / Apple,勒索软件的**物理本质一致**:

1. **密集加密** → 加密扩展/SIMD 指令占比升高(指令集不同,但都有专用加密指令);
2. **全盘流式遍历** → 各级缓存未命中(L1/L2/LLC)飙升(冯诺依曼架构通用规律);
3. **大量密文写回** → store 占比与写带宽抬升;
4. **突发节奏** → 短时间内完成,行为呈脉冲式(区别于平稳的合法加密备份)。

本工程的**特征工程思想(比值归一化 + 时间差分 + 缓存/加密/节奏三维度)是跨平台
通用的**,只需把**原始 PMU 事件映射**换成目标平台:

| 平台 | 加密指令事件 | 缓存事件 | 采集接口 |
|------|------------|---------|---------|
| x86 Windows | AES-NI 相关(Intel/AMD PMU) | LLC/L2/L1 miss | ETW / Intel PCM / perf |
| ARM Linux | ARMv8 `crypto_spec`(0x77) | `ll_cache_miss` 等 | `perf stat -e -I` |
| Apple Silicon | Apple 私有事件(需映射) | Apple 私有缓存事件 | `kpc`/Instruments |

## 六、纯 PMU 方案的共同局限

三平台都一样:**纯硬件信号难分"勒索全盘加密"与"合法加密备份/全盘加密初始化"**
(见 [dataset.md](dataset.md) 的 encbackup 干扰项分析)。生产级检测需补充:
- **文件层**:写入文件的熵变化(低熵→高熵)、扩展名批量改写、卷影副本删除;
- **身份层**:进程签名/公证、白名单、已知备份/AV 进程豁免;
- **行为序列**:勒索特有的"删备份 → 加密 → 投递勒索信"序列。

PMU 检测的定位:**早期、轻量、难规避的硬件级初筛**,与上述信号融合才构成完整防线。

## 七、结论

- **x86 Windows**:威胁最大,特征思想可迁移但需换 x86 PMU 事件与 AES-NI 计数器;
- **ARM**:本工程的**原生平台**,迁移成本最低,是增长最快的新战场(NAS/云/虚拟化);
- **Apple Silicon**:防御门槛高、威胁在增长,PMU 私有 + 硬件 AES 引擎使加密指令信号
  减弱,需更依赖缓存遍历与节奏特征,采集工程难度最大。

本工程当前实现面向 **ARMv8-A**,作为跨平台 PMU 勒索检测的**原型与方法论验证**;
迁移到 x86/Apple 主要是**事件映射 + 采集接口**的工程适配,核心特征设计通用。
