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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/ReinterpretCastSinking.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "ReinterpretCastSinking";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace mlir::triton::CVSplit {

void ReinterpretCastSinkingPass::runOnOperation() {
  ModuleOp mod = getOperation();
  DominanceInfo domInfo;

  // Collect all reinterpret_cast ops first, since we'll be modifying the IR.
  SmallVector<memref::ReinterpretCastOp> opsToProcess;
  mod->walk([&](memref::ReinterpretCastOp op) { opsToProcess.push_back(op); });

  int totalProcessed = 0;
  int totalCloned = 0;

  for (auto reinterpretOp : opsToProcess) {
    Value result = reinterpretOp.getResult();
    Block *defBlock = reinterpretOp->getBlock();

    // Group users by their parent block.
    DenseMap<Block *, SmallVector<Operation *>> blockUsers;
    for (auto *user : result.getUsers()) {
      blockUsers[user->getBlock()].push_back(user);
    }

    // If all uses are in the defining block, nothing to do.
    if (blockUsers.size() == 1 && blockUsers.count(defBlock))
      continue;

    LOG_DEBUG("Processing " << reinterpretOp << " with " << blockUsers.size()
                            << " target blocks");

    // For each target block different from defBlock, clone the op.
    OpBuilder builder(reinterpretOp->getContext());
    bool anyCloned = false;

    for (auto &[targetBlock, users] : blockUsers) {
      if (targetBlock == defBlock)
        continue;

      // Find the earliest user in this block as insertion point.
      Operation *firstUser = nullptr;
      for (auto &op : *targetBlock) {
        for (auto *user : users) {
          if (user == &op) {
            firstUser = user;
            break;
          }
        }
        if (firstUser)
          break;
      }

      if (!firstUser) {
        LOG_DEBUG("  Could not find first user in target block");
        continue;
      }

      // Check that all operands dominate the insertion point.
      bool operandsDominate = true;
      for (Value operand : reinterpretOp->getOperands()) {
        if (auto *operandOp = operand.getDefiningOp()) {
          if (!domInfo.dominates(operandOp, firstUser)) {
            LOG_DEBUG("  Operand " << operand << " does not dominate target");
            operandsDominate = false;
            break;
          }
        } else {
          // Block arguments always dominate within their region.
          auto *operandBlock = operand.getParentBlock();
          if (!domInfo.dominates(operandBlock, targetBlock)) {
            LOG_DEBUG("  Block arg " << operand
                                     << " block does not dominate target");
            operandsDominate = false;
            break;
          }
        }
      }

      if (!operandsDominate) {
        LOG_DEBUG("  Skipping target block due to dominance");
        continue;
      }

      // Clone the reinterpret_cast before the first user.
      builder.setInsertionPoint(firstUser);
      auto *clonedOp = builder.clone(*reinterpretOp.getOperation());
      Value clonedResult = clonedOp->getResult(0);

      LOG_DEBUG("  Cloned before " << *firstUser << " in block");

      // Redirect all users in this block to the cloned result.
      for (auto *user : users) {
        user->replaceUsesOfWith(result, clonedResult);
      }

      anyCloned = true;
      totalCloned++;
    }

    // If no uses remain in the defining block, erase the original.
    if (anyCloned && result.use_empty()) {
      LOG_DEBUG("  Erasing original: " << reinterpretOp);
      reinterpretOp->erase();
    }

    totalProcessed++;
  }

  LOG_DEBUG("Processed " << totalProcessed << " ops, cloned " << totalCloned
                         << " times");
}

std::unique_ptr<OperationPass<ModuleOp>> createReinterpretCastSinkingPass() {
  return std::make_unique<ReinterpretCastSinkingPass>();
}

} // namespace mlir::triton::CVSplit
