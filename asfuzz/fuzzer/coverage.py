from __future__ import annotations

from collections import Counter
import math

from asfuzz.spec.opspec import OpSpec


class CoverageTracker:
    def __init__(self) -> None:
        self.op_counts: Counter[str] = Counter()
        self.dtype_counts: Counter[str] = Counter()
        self.rank_counts: Counter[int] = Counter()
        self.work_bucket_counts: Counter[str] = Counter()
        self.shape_feature_counts: Counter[str] = Counter()
        self.signature_counts: Counter[str] = Counter()

    def add(self, spec: OpSpec) -> None:
        self.op_counts[spec.op_kind] += 1
        self.dtype_counts[spec.dtype()] += 1
        output = spec.tensors_by_role("output")[0]
        self.rank_counts[len(output.axes)] += 1
        self.work_bucket_counts[work_bucket(work_items(spec))] += 1
        for feature in shape_features(spec):
            self.shape_feature_counts[feature] += 1
        self.signature_counts[spec.signature()] += 1

    def diversity_score(self, spec: OpSpec) -> float:
        output = spec.tensors_by_role("output")[0]
        score = 0.0
        score += 0.35 / (1.0 + self.op_counts[spec.op_kind])
        score += 0.05 / (1.0 + self.dtype_counts[spec.dtype()])
        score += 0.15 / (1.0 + self.rank_counts[len(output.axes)])
        score += 0.20 / (1.0 + self.work_bucket_counts[work_bucket(work_items(spec))])
        features = shape_features(spec)
        feature_score = sum(1.0 / (1.0 + self.shape_feature_counts[feature]) for feature in features)
        score += 0.25 * feature_score / max(1, len(features))
        return score

    def to_dict(self) -> dict:
        return {
            "op_counts": dict(self.op_counts),
            "dtype_counts": dict(self.dtype_counts),
            "rank_counts": dict(self.rank_counts),
            "work_bucket_counts": dict(self.work_bucket_counts),
            "shape_feature_counts": dict(self.shape_feature_counts),
            "unique_signatures": len(self.signature_counts),
        }


def work_items(spec: OpSpec) -> int:
    total = 0
    for output in spec.tensors_by_role("output"):
        elems = 1
        for dim in spec.shape_of(output):
            elems *= dim
        total += elems
    total = max(1, total)
    for axis in spec.axes.values():
        if axis.is_reduce:
            total *= axis.size
    return int(total)


def work_bucket(items: int) -> str:
    if items <= 0:
        return "0"
    power = int(math.log2(items))
    lower = 1 << power
    upper = (1 << (power + 1)) - 1
    return f"{lower}-{upper}"


def complexity_score(spec: OpSpec, max_work_items: int) -> float:
    if max_work_items <= 1:
        return 0.0
    score = math.log2(max(1, work_items(spec))) / math.log2(max_work_items)
    if spec.op_kind in {"conv2d", "softmax", "reduce"}:
        score += 0.08
    if any(axis.size in {1, 3, 7, 13, 31, 63, 65, 127, 129} for axis in spec.axes.values()):
        score += 0.06
    if len(spec.tensors_by_role("output")[0].axes) >= 4:
        score += 0.05
    return max(0.0, min(1.0, score))


def shape_features(spec: OpSpec) -> list[str]:
    sizes = [axis.size for axis in spec.axes.values()]
    rank = len(spec.tensors_by_role("output")[0].axes)
    features = [f"rank:{rank}", f"op:{spec.op_kind}"]
    if any(size == 1 for size in sizes):
        features.append("has_unit_dim")
    if any(size in {3, 7, 13, 31, 63, 65, 127, 129} for size in sizes):
        features.append("has_odd_primeish_dim")
    if any(size % 16 == 0 for size in sizes):
        features.append("has_vector_dim")
    if any(size > 64 for size in sizes):
        features.append("has_large_dim")
    if spec.epilogue:
        features.append("has_epilogue")
    if spec.op_kind == "conv2d":
        features.append(f"conv_stride:{spec.extra.get('stride')}")
        features.append(f"conv_pad:{spec.extra.get('pad')}")
        features.append(f"conv_dilation:{spec.extra.get('dilation', 1)}")
    if spec.op_kind in {"reduce", "softmax"}:
        features.append(f"reduce_axis:{spec.extra.get('axis')}")
    if spec.op_kind in {"elementwise", "unary", "reduce"}:
        features.append(f"inner_op:{spec.extra.get('op')}")
    return features
