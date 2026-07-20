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

#ifndef ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#define ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <string_view>

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"

namespace mlir {
namespace CVPipeline {

inline constexpr llvm::StringLiteral kCoreType = "ssbuffer.core_type";
inline constexpr llvm::StringLiteral kBlockId = "ssbuffer.block_id";
inline constexpr llvm::StringLiteral kTransferId = "ssbuffer.transfer_id";
inline constexpr llvm::StringLiteral kCubeFirst = "ssbuffer.cube_first";
inline constexpr llvm::StringLiteral kVectorFirst = "ssbuffer.vector_first";
inline constexpr llvm::StringLiteral kAddFromMatmul =
    "ssbuffer.add_from_matmul";
inline constexpr llvm::StringLiteral kMainLoop = "ssbuffer.main_loop";
inline constexpr llvm::StringLiteral kTcoreType = "hivm.tcore_type";
inline constexpr llvm::StringLiteral kIf = "ssbuffer.if";
inline constexpr llvm::StringLiteral kIntraBuffer = "ssbuffer.intra_buffer";
inline constexpr llvm::StringLiteral kIntraBufCount =
    "ssbuffer.intra_buf_count";
inline constexpr llvm::StringLiteral kInterCoreBufCount =
    "ssbuffer.inter_core_buf_count";
inline constexpr llvm::StringLiteral kLoadStoreBufCount =
    "ssbuffer.load_store_buf_count";
inline constexpr llvm::StringLiteral kAnalyzeFlagId =
    "ssbuffer.analyze_flag_id";
inline constexpr llvm::StringLiteral kLoopCarriedL0C =
    "ssbuffer.loop_carried_l0c";
inline constexpr llvm::StringLiteral kCrossCoreDeps = "ssbuffer.crossCoreDeps";
inline constexpr llvm::StringLiteral kIntraDeps = "ssbuffer.intraDeps";
inline constexpr llvm::StringLiteral kMemCrossDeps = "ssbuffer.memCrossDeps";
inline constexpr llvm::StringLiteral kMayNotExec = "ssbuffer.may_not_exec";
inline constexpr llvm::StringLiteral kClone = "ssbuffer.clone";
inline constexpr llvm::StringLiteral kEnableUbRefineOpt =
    "ssbuffer.enable_ub_refine_opt";
inline constexpr llvm::StringLiteral kInsertionOptimization =
    "ssbuffer.insertionOptimization";
static constexpr llvm::StringLiteral kInlinableQuantScaleAttr =
    "enable_fast_tf32_mul";
inline constexpr llvm::StringLiteral kHIVMMatmulLimitedInCubeAttr =
    "hivm.matmul_limited_in_cube";

inline constexpr const char *ERRCODE_ATTR =
    "triton_ascend.dynamic_cv_pipeline.rc";
static constexpr const int ERRCODE_FAILED = 1;
static constexpr const int ERRCODE_IGNORED = 2;
constexpr int64_t CACHE_TABLE_BUFFER_SIZE = 4096;
constexpr int64_t BYTE_SIZE = 8;
static constexpr int crossCoreProducerId = 1;
static constexpr int crossCoreConsumerId = 0;

enum CoreType {
  UNDETERMINED = 0,
  VECTOR_ONLY = 1 << 0,
  CUBE_ONLY = 1 << 1,
  CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY,
};

inline constexpr CoreType fromStrCoreType(std::string_view s) {
  if (s == "VECTOR") {
    return CoreType::VECTOR_ONLY;
  }
  if (s == "CUBE") {
    return CoreType::CUBE_ONLY;
  }

  return CoreType::UNDETERMINED;
}

void setEnableCubeBlockMerge(bool enable);
bool isCubeBlockMergeEnabled();

void setEnableUBRefineOpt(bool enable);
bool isUBRefineOptEnabled();

// Functions for managing core types
CoreType getOpCoreType(Operation *op);
std::optional<int> getOpBlockId(Operation *op);
llvm::LogicalResult verifyOpBlockId(Operation *op);
int getAvailableBlockId(ModuleOp module);
void setFallbackAttr(ModuleOp module, int errorCode);
bool hasFallbackAttr(ModuleOp module);
bool isScfOp(Operation *op);
bool isOnlyDirectlyUse(Operation *preOp, Operation *nextOp,
                       const CVPipeline::MemoryDependenceGraph &memGraph);

// Wrapper around a "main loop" — either scf.for or scf.while carrying the
// ssbuffer.main_loop attribute. Lets downstream code treat both uniformly.
class MainLoop {
public:
  Operation *op = nullptr;
  Block *body = nullptr;
  Value iterCounter;

  Block *getBody() const { return body; }
  Operation *getOperation() const { return op; }
  MLIRContext *getContext() const { return op->getContext(); }
  Location getLoc() const { return op->getLoc(); }
  Block *getBlock() const { return op->getBlock(); }
  Block::iterator getIterator() const { return op->getIterator(); }
  Operation *operator->() const { return op; }
  bool isWhile() const { return isa<scf::WhileOp>(op); }

  // Ordered list of iter args carried across loop iterations, as the
  // region's BlockArguments (same Value identity as forOp.getRegionIterArgs()).
  // forOp: getRegionIterArgs(). whileOp: before-body args (== after-body args).
  SmallVector<Value> getIterArgs() const {
    SmallVector<Value> result;
    if (auto f = dyn_cast<scf::ForOp>(op))
      result.append(f.getRegionIterArgs().begin(), f.getRegionIterArgs().end());
    else if (auto w = dyn_cast<scf::WhileOp>(op)) {
      Block::BlockArgListType args = w.getBeforeBody()->getArguments();
      result.append(args.begin(), args.end());
    }
    return result;
  }

  static MainLoop get(Operation *o) {
    MainLoop ml;
    ml.op = o;
    if (auto f = dyn_cast<scf::ForOp>(o))
      ml.body = f.getBody();
    else if (auto w = dyn_cast<scf::WhileOp>(o))
      ml.body = w.getAfterBody();
    return ml;
  }
};

// True when `op` is a main_loop loop op (forOp / whileOp carrying the tag).
inline bool isMainLoopOp(Operation *op) {
  return op && isa<scf::ForOp, scf::WhileOp>(op) && op->hasAttr(kMainLoop);
}

inline bool isCubeOp(Operation *op) {
  return !isScfOp(op) && CVPipeline::getOpCoreType(op) == CoreType::CUBE_ONLY;
}

bool isVectorOnlyOp(Operation *op);

bool isScalarLike(Value value);
bool isStoreLike(Operation *op);
bool isViewLike(Operation *op);

} // namespace CVPipeline
} // namespace mlir

#endif
