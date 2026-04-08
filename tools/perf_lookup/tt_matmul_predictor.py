import sys
import os
import math
import re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from step2_predict_config import predict_op
from step3_predict_time   import predict_time

TILE = 32

def pad_to_tile(x):
    return ((x + TILE - 1) // TILE) * TILE


class MatmulPredictorStats:
    def __init__(self, msecs):
        self.msecs            = msecs
        self.matrix_pipe_util = 60.0
        self.vector_pipe_util = 0.0
        self.memory_traffic   = None
        self.mem_util         = None


def _shape_from_tensor(t):
    """
    Safely convert a tensor's shape to a Python list.
    Handles Shape objects, _shape attributes, iterables.
    Same logic as existing LUT (_coerce_shape_like_to_list).
    """
    for attr in ('shape', 'logical_shape', '_shape'):
        s = getattr(t, attr, None)
        if s is None:
            continue
        # Shape object with .view()
        if hasattr(s, 'view') and callable(s.view):
            try:
                return [int(x) for x in s.view()]
            except Exception:
                pass
        # Shape object with ._shape
        if hasattr(s, '_shape'):
            try:
                return [int(x) for x in s._shape]
            except Exception:
                pass
        # direct list/tuple
        if isinstance(s, (list, tuple)):
            return [int(x) for x in s]
        # try iterating
        try:
            return [int(x) for x in s]
        except Exception:
            pass
    return None


# ── Method 1: wlgraph._tensors dict (same as existing LUT) ───────────────────

def _shapes_from_wlgraph_tensors(op, wlgraph):
    """
    Get shapes from wlgraph._tensors dict.
    This is exactly how the existing LUT works:
      tensors = wlgraph._tensors  (dict: name → tensor object)
      tensor  = tensors[inList[0]]
      shape   = tensor.shape._shape
    Most reliable method — no math, no assumptions.
    """
    try:
        tensors = getattr(wlgraph, '_tensors', {})
        inList  = getattr(op, 'inList', [])

        if not tensors or len(inList) < 2:
            return None, None

        t_a = tensors.get(inList[0])
        t_b = tensors.get(inList[1])

        if t_a is None or t_b is None:
            return None, None

        sa = _shape_from_tensor(t_a)
        sb = _shape_from_tensor(t_b)

        if sa is not None and sb is not None:
            return sa, sb

    except Exception:
        pass
    return None, None


# ── Method 2: wlgraph.get_op() attributes ────────────────────────────────────

def _shapes_from_wlgraph_op(op, wlgraph):
    """
    Get shapes from wlgraph op attributes.
    Tries input_tensors string, iTList, inList via get_op.
    """
    try:
        op_from_graph = wlgraph.get_op(op.name)

        # try input_tensors string attribute
        for attr in ['input_tensors', 'inputTensors', 'in_tensors']:
            s = getattr(op_from_graph, attr, None)
            if s is not None:
                matches = re.findall(r'\[([0-9x]+)\]', str(s))
                if len(matches) >= 2:
                    return (
                        [int(x) for x in matches[0].split('x')],
                        [int(x) for x in matches[1].split('x')]
                    )

        # try iTList
        iTList = getattr(op_from_graph, 'iTList', [])
        if len(iTList) >= 2:
            sa = _shape_from_tensor(iTList[0])
            sb = _shape_from_tensor(iTList[1])
            if sa is not None and sb is not None:
                return sa, sb

    except Exception:
        pass

    # try input_tensors on original op
    for attr in ['input_tensors', 'inputTensors', 'in_tensors']:
        s = getattr(op, attr, None)
        if s is not None:
            matches = re.findall(r'\[([0-9x]+)\]', str(s))
            if len(matches) >= 2:
                return (
                    [int(x) for x in matches[0].split('x')],
                    [int(x) for x in matches[1].split('x')]
                )

    return None, None


# ── Method 3: inList tensor names ────────────────────────────────────────────

def _shapes_from_inlist_names(op, wlgraph):
    """
    Parse shapes from inList tensor name strings.
    Works when tensor names contain shape like [8x197x768].
    """
    inList = getattr(op, 'inList', [])
    if len(inList) < 2:
        return None, None
    try:
        matches_a = re.findall(r'\[([0-9x]+)\]', str(inList[0]))
        matches_b = re.findall(r'\[([0-9x]+)\]', str(inList[1]))
        if matches_a and matches_b:
            return (
                [int(x) for x in matches_a[0].split('x')],
                [int(x) for x in matches_b[0].split('x')]
            )
    except Exception:
        pass
    return None, None


# ── Method 4: perf_stats math (last resort) ──────────────────────────────────

def _shapes_from_perf_stats(op):
    """
    Derive shapes from MAC count math.
    NOTE: inActCount excludes weights so quadratic often fails.
    Use only as last resort.

    mac      = batch * M * K * N
    outElems = batch * M * N
    K = mac / outElems
    Solve quadratic for M, N, batch.
    """
    perf     = getattr(op, 'perf_stats', {})
    mac      = perf.get('instrs', {}).get('mac', 0)
    inElems  = perf.get('inActCount', perf.get('inElems', 0))
    outElems = perf.get('outActCount', perf.get('outElems', 0))

    if mac == 0 or outElems == 0 or inElems == 0:
        return None, None
    if mac % outElems != 0:
        return None, None

    K = mac // outElems
    if K == 0:
        return None, None
    if inElems % K != 0:
        return None, None

    BM_plus_N = inElems // K
    BMN       = outElems
    disc      = BM_plus_N * BM_plus_N - 4 * BMN
    if disc < 0:
        return None, None

    sqrt_disc = int(math.isqrt(disc))
    if sqrt_disc * sqrt_disc != disc:
        return None, None

    # two solutions — pick the one where N >= M (larger dim is N)
    # and batch is a reasonable power of 2
    candidates = []
    for y in [(BM_plus_N + sqrt_disc) // 2,
              (BM_plus_N - sqrt_disc) // 2]:
        if y <= 0 or BMN % y != 0:
            continue
        x = BMN // y
        if x <= 0:
            continue
        # find batch
        for b in sorted([1,2,4,6,8,12,16,24,32,48,64,96,112,128,256],
                        reverse=True):
            if x % b == 0 and x // b >= 1:
                candidates.append((b, x // b, K, y))  # batch, M, K, N
                break

    if not candidates:
        return None, None

    # prefer solution where batch is small (8 for ViT) and M is large
    # sort by: prefer batch<=128, prefer M divisible by 32
    def score(c):
        batch, M, K, N = c
        return (batch > 128, M % 32 != 0, batch, -M)

    candidates.sort(key=score)
    batch, M, K_val, N = candidates[0]

    return [batch, M, K_val], [K_val, N]


# ── Main predictor ─────────────────────────────────────────────────────────────

def predict_matmul_msecs(op, wlgraph, freq_MHz=1000.0):

    optype = getattr(op, 'optype', '') or ''
    if 'matmul' not in optype.lower() and 'MatMul' not in optype:
        return None

    if getattr(op, 'fused_in_optimization', False):
        return None

    try:
        # method 1: wlgraph._tensors dict (same as existing LUT)
        shape_a, shape_b = _shapes_from_wlgraph_tensors(op, wlgraph)

        # method 2: wlgraph op attributes
        if shape_a is None:
            shape_a, shape_b = _shapes_from_wlgraph_op(op, wlgraph)

        # method 3: inList tensor names
        if shape_a is None:
            shape_a, shape_b = _shapes_from_inlist_names(op, wlgraph)

        # method 4: perf_stats math (last resort)
        if shape_a is None:
            shape_a, shape_b = _shapes_from_perf_stats(op)

        # debug for first 3 failing ops
        if shape_a is None:
            if not hasattr(predict_matmul_msecs, '_debug_count'):
                predict_matmul_msecs._debug_count = 0
            if predict_matmul_msecs._debug_count < 3:
                predict_matmul_msecs._debug_count += 1
                perf = getattr(op, 'perf_stats', {})
                print(f"\n[NONE DEBUG] op={op.name}")
                print(f"  inList={getattr(op, 'inList', None)}")
                print(f"  mac={perf.get('instrs',{}).get('mac',0)}")
                print(f"  inElems={perf.get('inActCount', perf.get('inElems',0))}")
                print(f"  outElems={perf.get('outActCount', perf.get('outElems',0))}")
                tensors = getattr(wlgraph, '_tensors', {})
                print(f"  wlgraph._tensors keys (first 5): {list(tensors.keys())[:5]}")
                inList = getattr(op, 'inList', [])
                if inList:
                    print(f"  tensors.get(inList[0]) = {tensors.get(inList[0])}")
            return None

        if len(shape_a) < 2 or len(shape_b) < 2:
            return None

        # extract M, K, N, batch
        M     = shape_a[-2]
        K     = shape_a[-1]
        N     = shape_b[-1]
        batch = 1
        for d in shape_a[:-2]:
            batch *= d
        batch = max(1, batch)

        # pad to tile boundary — matches hardware
        M = pad_to_tile(M)
        K = pad_to_tile(K)
        N = pad_to_tile(N)

        if M == 0 or K == 0 or N == 0:
            return None

        print(f"[PREDICTOR] {op.name}  batch={batch} M={M} K={K} N={N}")

        raw_op = dict(
            op_id       = 0,
            description = f"M{M}_K{K}_N{N}_b{batch}",
            batch       = batch,
            M           = M, K = K, N = N,
            batch_b     = 1,
            layout_a    = "INTERLEAVED", buffer_a  = "DRAM",
            layout_b    = "INTERLEAVED", buffer_b  = "DRAM",
            layout_out  = "INTERLEAVED", buffer_out = "DRAM",
        )

        configured_op = predict_op(raw_op)
        if configured_op.get("config_type") in (None, "UNKNOWN", "ERROR"):
            return None

        timed_op = predict_time(configured_op)
        pred_ns  = timed_op.get("predicted_ns")

        if not pred_ns or pred_ns <= 0:
            return None

        print(f"[PREDICTOR] {op.name} → {pred_ns:,} ns  ({pred_ns/1e6:.3f} ms)")

        return MatmulPredictorStats(msecs=pred_ns / 1e6)

    except Exception as e:
        import traceback
        print(f"[PREDICTOR ERROR] {op.name}: {e}")
        print(traceback.format_exc())
        return None