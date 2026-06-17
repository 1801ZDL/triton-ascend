# AddMultiBufferOuterScope: WhileOp 适配设计

## 1. 背景

`AddMultiBufferOuterScope` pass 负责核间（VECTOR↔CUBE）多 buffer 优化，核心分三步：
1. 识别传输对并打标签（crossDeps）
2. 构建双 buffer 结构（output alloc + mark + sync）
3. 添加轮询控制流（scf.if 包裹）

改造前仅支持 `scf::ForOp`，需要适配 `scf::WhileOp`。

## 2. ForOp vs WhileOp 关键差异

| 特性 | scf::ForOp | scf::WhileOp |
|------|-----------|-------------|
| 迭代变量 | `getInductionVar()` / `getStep()` | 无 |
| Region 结构 | 单一 body region | `before` + `after` 两个 region |
| 终止 op | `scf.yield` | `scf.condition`(before) / `scf.yield`(after) |

## 3. 核心修改

### 3.1 main_loop 属性识别泛化

```cpp
// 改造前: 仅识别 ForOp
static bool forOpHasMainLoopAttr(scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) return true;
    Operation *term = forOp.getBody()->getTerminator();
    return term && term->hasAttr("ssbuffer.main_loop");
}

// 改造后: 通用识别，同时兼容 ForOp / WhileOp 的 terminator 属性
static bool loopOpHasMainLoopAttr(Operation *op) {
    if (op->hasAttr("ssbuffer.main_loop")) return true;
    // ForOp: main_loop 可能在 scf.yield 上
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto *term = forOp.getBody()->getTerminator();
        return term && term->hasAttr("ssbuffer.main_loop");
    }
    // WhileOp: main_loop 可能在 scf.condition 或 scf.yield 上
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        if (auto *t = whileOp.getBefore().front().getTerminator())
            if (t->hasAttr("ssbuffer.main_loop")) return true;
        if (auto *t = whileOp.getAfter().front().getTerminator())
            if (t->hasAttr("ssbuffer.main_loop")) return true;
    }
    return false;
}
```

`parentOpHasMainLoopAttr` 同步扩展，增加 `isa<scf::ForOp, scf::WhileOp>` 类型过滤：

```cpp
static bool parentOpHasMainLoopAttr(Operation *syncOp) {
    Operation *parent = syncOp->getParentOp();
    if (!parent) return false;
    if (isa<scf::ForOp, scf::WhileOp>(parent))
        return loopOpHasMainLoopAttr(parent);
    return false;
}
```

### 3.2 轮询条件 — WhileOp 的 toggle 注入方案

ForOp 有 induction variable，可以直接计算 `(iv / step) % 2 == 0` 作为轮询条件。
WhileOp 没有，因此采用 **i1 toggle 注入** 方案：

**预处理阶段**（Step 1 之前执行）：遍历所有带 `main_loop` 的 WhileOp，
向其注入一个 `i1` loop-carried 变量，每次迭代 `xori` 翻转。

```
输入:
  scf.while (%args = %inits) : (T...) -> (T...) {
    scf.condition(%cond) %args : T...
  } do {
  ^bb0(%args: T...):
    ...
    scf.yield %new_args : T...
  }

输出:
  %false = arith.constant false
  scf.while (%args = %inits, %toggle = %false) : (T..., i1) -> (T..., i1) {
    scf.condition(%cond) %args, %toggle : T..., i1
  } do {
  ^bb0(%args: T..., %toggle: i1):
    ...
    %true = arith.constant true
    %next = arith.xori %toggle, %true : i1
    scf.yield %new_args, %next : T..., i1
  }
```

核心函数 `ensureWhileOpHasToggle`：
- 检查 `ssbuffer.polling_toggle` 属性，避免重复注入
- 克隆 before/after region，追加 toggle 到 block args / condition / yield
- 拷贝原 WhileOp 的 `ssbuffer.*` 属性（尤其是 `main_loop`）
- 替换旧 WhileOp，返回 after block 的 toggle arg 作为轮询条件

预处理必须在 Step 1 之前执行，否则后续数据收集持有的 op 指针会在 WhileOp 被 erase 后变为 dangling。

### 3.3 轮询条件分发

`prepareLoopPolling` 统一 ForOp / WhileOp 的轮询条件获取：

```cpp
static Value prepareLoopPolling(Operation *loopOp, Operation *waitOp,
                                OpBuilder &builderOut) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
        // ForOp: (iv / step) % 2 == 0
        OpBuilder condBuilder(forOp.getBody(), Block::iterator(waitOp));
        Value cond = createPollingCondition(forOp, condBuilder, ...);
        builderOut.setInsertionPoint(forOp.getBody()->getTerminator());
        return cond;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
        // WhileOp: toggle 已在预处理中注入，直接取 after block 最后一个 arg
        Block &after = whileOp.getAfter().front();
        Value toggleArg = after.getArgument(after.getNumArguments() - 1);
        builderOut.setInsertionPoint(after.getTerminator());
        return toggleArg;
    }
    llvm_unreachable("unexpected loop op type");
}
```

### 3.4 poll 控制流 — `addPollingControlFlow` 改造

改造前硬转型 `cast<scf::ForOp>(senderWaitParent)`。
改造后通过 `prepareLoopPolling` 统一分发，消除对具体 loop 类型的依赖。

```cpp
// 改造前
scf::ForOp senderForOp = cast<scf::ForOp>(senderWaitParent);
Value senderCond = createPollingCondition(senderForOp, ...);

// 改造后
OpBuilder senderBuilder(...);
Value senderCond = prepareLoopPolling(senderWaitParent,
                                       g.senderChain.waitOp, senderBuilder);
```

`processTransferChain` 及其内部的 `wrapSyncOpWithScfIf` / `wrapTransferOpWithScfIf*`
只消费 `Value cond`，不感知 cond 的来源类型，**无需修改**。

## 4. 修改文件清单

| 文件 | 改动 |
|------|------|
| `AddMultiBufferOuterScope.cpp` | `loopOpHasMainLoopAttr` 泛化、`ensureWhileOpHasToggle` 新增、`prepareLoopPolling` 新增、`preInjectWhileOpToggles` 预处理、`addPollingControlFlow` 去硬转型 |
| `Outer-scope-whileop.mlir` | 新增 UT：C→V sender 在 while、双端 while 场景 |

## 5. 测试验证

- 单 buffer 模式：crossDeps 标签正确，pass 不 crash
- 双 buffer 模式：toggle 注入 + output buffer 创建 + scf.if 轮询包裹 + TCB 标签配对，均通过
