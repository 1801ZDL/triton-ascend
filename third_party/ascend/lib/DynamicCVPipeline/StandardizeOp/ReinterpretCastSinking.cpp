/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "llvm/Support/Debug.h"
#include <memory>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/ReinterpretCastSinking.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "ReinterpretCastSinking";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

// Find the earliest user of `result` that lives in `block`, scanning the
// block in IR order. Used as the clone insertion point for that block.
Operation *findFirstUserInBlock(ArrayRef<Operation *> users, Block *block) {
  for (auto &op : *block) {
    for (auto *user : users) {
      if (user == &op)
        return user;
    }
  }
  return nullptr;
}

// Sink one reinterpret_cast into child use blocks:
// 1. Group users by their parent block.
// 2. Clone before each target block's first user.
// 3. Redirect that block's users to the clone.
// 4. Erase original if no remaining uses.
int sinkReinterpretCastOp(memref::ReinterpretCastOp reinterpretOp) {
  Value result = reinterpretOp.getResult();
  Block *defBlock = reinterpretOp->getBlock();

  // Group users by their parent block.
  DenseMap<Block *, SmallVector<Operation *>> blockUsers;
  for (auto *user : result.getUsers()) {
    blockUsers[user->getBlock()].push_back(user);
  }

  // If all uses are in the defining block, nothing to do.
  if (blockUsers.size() == 1 && blockUsers.count(defBlock)) {
    return 0;
  }

  LOG_DEBUG("Processing " << reinterpretOp << " with " << blockUsers.size()
                          << " target blocks");

  OpBuilder builder(reinterpretOp->getContext());
  int cloned = 0;

  for (auto &[targetBlock, users] : blockUsers) {
    if (targetBlock == defBlock) {
      continue;
    }

    Operation *firstUser = findFirstUserInBlock(users, targetBlock);
    if (!firstUser) {
      LOG_DEBUG("  Could not find first user in target block");
      continue;
    }

    // Clone the reinterpret_cast before the first user.
    builder.setInsertionPoint(firstUser);
    Value clonedResult =
        builder.clone(*reinterpretOp.getOperation())->getResult(0);
    LOG_DEBUG("  Cloned before " << *firstUser << " in block");

    // Redirect all users in this block to the cloned result.
    for (auto *user : users) {
      user->replaceUsesOfWith(result, clonedResult);
    }

    ++cloned;
  }

  // If no uses remain in the defining block, erase the original.
  if (cloned > 0 && result.use_empty()) {
    LOG_DEBUG("  Erasing original: " << reinterpretOp);
    reinterpretOp->erase();
  }

  return cloned;
}

} // namespace

namespace mlir::triton::CVSplit {

void ReinterpretCastSinkingPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // Collect all reinterpret_cast ops first, since we'll be modifying the IR.
  SmallVector<memref::ReinterpretCastOp> opsToProcess;
  mod->walk([&](memref::ReinterpretCastOp op) { opsToProcess.push_back(op); });

  int totalCloned = 0;
  for (auto reinterpretOp : opsToProcess) {
    totalCloned += sinkReinterpretCastOp(reinterpretOp);
  }

  LOG_DEBUG("Processed " << opsToProcess.size() << " ops, cloned "
                         << totalCloned << " times");
}

std::unique_ptr<OperationPass<ModuleOp>> createReinterpretCastSinkingPass() {
  return std::make_unique<ReinterpretCastSinkingPass>();
}

} // namespace mlir::triton::CVSplit
