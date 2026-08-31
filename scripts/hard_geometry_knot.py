#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#  hard_geometry_knot.py
#
#  Generates a watertight, orientable, self-intersection-free triangle mesh
#  meant as a worst case input for sphere / ball packing algorithms.
#
#  The shape is a single closed tube swept along a (3,2) torus knot.  The cross
#  section morphs along the loop between:
#
#      * round bulges          -> largest local feature size
#      * slender rods          -> aspect ratio ~200:1
#      * thin ribbons          -> shells with two knife edges
#      * 4 thin radial fins    -> plates + re-entrant corners
#      * 5- and 7-point stars  -> sharp convex tips, sharp concave valleys
#
#  and at the four places where two strands of the knot pass closest, round
#  bulges are placed so that the crossings become sub-millimetre slots.
#
#  Because the mesh is a swept closed tube it is watertight *by construction*:
#  no booleans, no remeshing, no hole filling.  Topologically it is a torus
#  (genus 1, Euler characteristic 0), which is expected and not a defect.
#
#  Usage
#  -----
#  Blender GUI   : Scripting workspace -> New -> paste this file -> Run Script
#  Blender CLI   : blender --background --python hard_geometry_knot.py
#  Plain Python  : python3 hard_geometry_knot.py        (needs only numpy)
#
#  Set EXPORT_DIR below to also write .ply / .stl / .obj next to the mesh.
# -----------------------------------------------------------------------------

import os
import numpy as np

# ============================== parameters ===================================

N_ALONG = 1000          # stations along the knot
N_CROSS = 50            # vertices per cross section
#                         triangles = 2 * N_ALONG * N_CROSS  -> 100,000
#                         vertices  =     N_ALONG * N_CROSS  ->  50,000

TARGET_SIZE = 100.0     # longest bounding box edge of the result (e.g. mm).
#                         Set to None to keep the native size (~3.5 units).

KNOT = dict(R=1.0, rt=0.42, P=3, Q=2)   # (P,Q) torus knot, gcd(P,Q) must be 1
TWISTS = 2.0            # extra full turns of the cross section over the loop
                        # (must be an integer so the tube closes on itself)

SLOT_GAP = 0.010        # native units. Minimum wall-to-wall clearance enforced
                        # between distinct strands. 0.010 -> ~0.3 mm at
                        # TARGET_SIZE = 100. Lower = harder, 0 = touching.
CURV_FRAC = 0.55        # radius <= CURV_FRAC / curvature. Keep below 1.0.

CONTACTS = (0.191, 0.303, 0.698, 0.797)   # arc fractions of closest approach
R_CONTACT = 0.352                         # bulge radius placed there
# NOTE: these four numbers belong to the default KNOT. If you change KNOT the
# script prints the new closest-approach fractions and warns you.

RUN_CHECKS = True               # topology / watertightness / normals report
RUN_SELF_INTERSECTION = True    # exact triangle-triangle test, ~10-20 s at 100k

OBJECT_NAME = "hard_geometry_knot"
EXPORT_DIR = ""                 # e.g. "/tmp" or "C:/tmp"; "" = do not export

TWO_PI = 2.0 * np.pi

# =========================== centreline: torus knot ==========================


def knot_pos(t, R, rt, P, Q):
    rad = R + rt * np.cos(Q * t)
    return np.stack([rad * np.cos(P * t), rad * np.sin(P * t), rt * np.sin(Q * t)], -1)


def knot_d1(t, R, rt, P, Q):
    a, b = Q * t, P * t
    rad = R + rt * np.cos(a)
    dr = -rt * Q * np.sin(a)
    return np.stack([dr * np.cos(b) - rad * P * np.sin(b),
                     dr * np.sin(b) + rad * P * np.cos(b),
                     rt * Q * np.cos(a)], -1)


def knot_d2(t, R, rt, P, Q):
    a, b = Q * t, P * t
    rad = R + rt * np.cos(a)
    dr = -rt * Q * np.sin(a)
    d2r = -rt * Q * Q * np.cos(a)
    return np.stack([d2r * np.cos(b) - 2 * dr * P * np.sin(b) - rad * P * P * np.cos(b),
                     d2r * np.sin(b) + 2 * dr * P * np.cos(b) - rad * P * P * np.sin(b),
                     -rt * Q * Q * np.sin(a)], -1)


def resample_arclength(n, kw, dense=200000):
    """Parameter values giving n points equally spaced in arc length."""
    t = np.linspace(0.0, TWO_PI, dense + 1)
    p = knot_pos(t, **kw)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return np.interp(np.linspace(0.0, cum[-1], n + 1)[:-1], cum, t), cum[-1]


# ================ rotation minimising frame + exact closure ==================


def _reflect_transport(p0, p1, t0, t1, u0):
    """One double-reflection step (Wang et al. 2008)."""
    v1 = p1 - p0
    c1 = v1 @ v1
    if c1 < 1e-30:
        return u0
    uL = u0 - (2.0 / c1) * (v1 @ u0) * v1
    tL = t0 - (2.0 / c1) * (v1 @ t0) * v1
    v2 = t1 - tL
    c2 = v2 @ v2
    u = uL if c2 < 1e-30 else uL - (2.0 / c2) * (v2 @ uL) * v2
    u = u - (u @ t1) * t1
    return u / np.linalg.norm(u)


def frames(pts, tan, twists):
    """Rotation minimising frame, corrected so ring N coincides with ring 0."""
    n = len(pts)
    U = np.zeros((n, 3))
    seed = np.array([0.0, 0.0, 1.0])
    if abs(seed @ tan[0]) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u = seed - (seed @ tan[0]) * tan[0]
    U[0] = u / np.linalg.norm(u)
    for i in range(n - 1):
        U[i + 1] = _reflect_transport(pts[i], pts[i + 1], tan[i], tan[i + 1], U[i])
    W = np.cross(tan, U)

    # transport once more onto ring 0 to measure the holonomy of the frame
    u = _reflect_transport(pts[-1], pts[0], tan[-1], tan[0], U[-1])
    delta = np.arctan2(u @ W[0], u @ U[0])

    s = np.arange(n) / n
    ang = -delta * s + TWO_PI * twists * s      # closes exactly at s = 1
    c, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return c * U + sn * W, -sn * U + c * W, delta


# ============================ cross sections =================================
# Every profile returns N_CROSS points in the local (u, v) plane, ordered
# counter-clockwise, index 0 on the +u axis, spacing close to uniform in arc
# length.  Profiles with corners keep their corners as exact vertices, so the
# sharp features stay sharp instead of being averaged away.


def _sample_smooth(polar, n, dense=8000):
    th = np.linspace(0.0, TWO_PI, dense + 1)
    r = polar(th)
    p = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    tq = np.interp(np.linspace(0.0, cum[-1], n + 1)[:-1], cum, th)
    r = polar(tq)
    return np.stack([r * np.cos(tq), r * np.sin(tq)], -1)


def _sample_polygon(corners, n):
    C = np.asarray(corners, float)
    m = len(C)
    if m > n:
        raise ValueError("N_CROSS too small for this profile (needs >= %d)" % m)
    Cn = np.roll(C, -1, axis=0)
    ln = np.linalg.norm(Cn - C, axis=1)
    ideal = n * ln / ln.sum()
    cnt = np.maximum(1, np.floor(ideal).astype(int))
    order = np.argsort(-(ideal - cnt))
    k, rem = 0, n - cnt.sum()
    while rem != 0:
        j = order[k % m]
        if rem > 0:
            cnt[j] += 1
            rem -= 1
        elif cnt[j] > 1:
            cnt[j] -= 1
            rem += 1
        k += 1
    out = []
    for i in range(m):
        a = np.linspace(0.0, 1.0, cnt[i] + 1)[:-1][:, None]
        out.append(C[i] * (1 - a) + Cn[i] * a)
    return np.concatenate(out, axis=0)


def prof_circle(n, r):
    return _sample_smooth(lambda th: np.full_like(th, r), n)


def prof_ribbon(n, half_width, half_thick):
    """Thin plate with a knife edge at each end."""
    a, b = half_width, half_thick
    corners = [(a, 0.0), (0.55 * a, b), (0.0, b), (-0.55 * a, b),
               (-a, 0.0), (-0.55 * a, -b), (0.0, -b), (0.55 * a, -b)]
    return _sample_polygon(corners, n)


def prof_star(n, r_in, r_out, k):
    """k sharp convex tips alternating with k sharp concave valleys."""
    corners = []
    for i in range(k):
        a0 = TWO_PI * i / k
        am = TWO_PI * (i + 0.5) / k
        corners += [(r_out * np.cos(a0), r_out * np.sin(a0)),
                    (r_in * np.cos(am), r_in * np.sin(am))]
    return _sample_polygon(corners, n)


def prof_fins(n, arm, half_thick, k=4):
    """k thin plates radiating from a small hub -> re-entrant corners."""
    corners = []
    for i in range(k):
        a0, a1 = TWO_PI * i / k, TWO_PI * (i + 1) / k
        ex = np.array([np.cos(a0), np.sin(a0)])
        ey = np.array([-np.sin(a0), np.cos(a0)])
        mid = np.array([np.cos(0.5 * (a0 + a1)), np.sin(0.5 * (a0 + a1))])
        corners += [tuple(arm * ex - half_thick * ey),
                    tuple(arm * ex),                    # vertex exactly on +u for i=0
                    tuple(arm * ex + half_thick * ey),
                    tuple(1.6 * half_thick * mid)]
    return corners[1:] + corners[:1]                    # rotate so index 0 is a tip


def default_schedule(n):
    """(arc fraction, label, profile) keyframes, blended smoothly in between."""
    sch = [(c, "bulge  round, forms a narrow slot with the facing strand",
            prof_circle(n, R_CONTACT)) for c in CONTACTS]
    return sch + [
        (0.000, "star5  5 sharp tips + 5 deep concave valleys", prof_star(n, 0.030, 0.320, 5)),
        (0.095, "rod    slender rod", prof_circle(n, 0.0090)),
        (0.245, "shell  thin ribbon with 2 knife edges", prof_ribbon(n, 0.320, 0.0050)),
        (0.400, "fins   4 thin plates, re-entrant corners", _sample_polygon(prof_fins(n, 0.300, 0.0050), n)),
        (0.500, "star7  7 sharp tips", prof_star(n, 0.035, 0.280, 7)),
        (0.600, "rod2   slender rod", prof_circle(n, 0.014)),
        (0.750, "shell2 thin ribbon", prof_ribbon(n, 0.280, 0.0040)),
        (0.890, "blob   thickest section", prof_circle(n, 0.360)),
    ]


# ================================ build ======================================


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _erode(x, w):
    n = len(x)
    pad = np.concatenate([x[-w:], x, x[:w]])
    return np.min(np.stack([pad[i:i + n] for i in range(2 * w + 1)]), 0)


def _movavg(x, w):
    n = len(x)
    pad = np.concatenate([x[-w:], x, x[:w]])
    return np.mean(np.stack([pad[i:i + n] for i in range(2 * w + 1)]), 0)


def build():
    n_a, n_c = N_ALONG, N_CROSS
    tt, L = resample_arclength(n_a, KNOT)
    pts = knot_pos(tt, **KNOT)
    d1, d2 = knot_d1(tt, **KNOT), knot_d2(tt, **KNOT)
    speed = np.linalg.norm(d1, axis=1)
    tan = d1 / speed[:, None]
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / speed ** 3
    U, W, delta = frames(pts, tan, TWISTS)
    ds = L / n_a

    # ---- blend the cross section keyframes along the loop --------------------
    sch = sorted(default_schedule(n_c), key=lambda z: z[0])
    pos = np.array([z[0] for z in sch])
    prof = np.stack([z[2] for z in sch])
    S = np.arange(n_a) / n_a
    i0 = np.searchsorted(pos, S, side="right") - 1
    lo = pos[i0]
    hi = np.where(np.append(pos, pos[0] + 1.0)[i0 + 1] > lo,
                  np.append(pos, pos[0] + 1.0)[i0 + 1],
                  np.append(pos, pos[0] + 1.0)[i0 + 1] + 1.0)
    u = ((S - lo) % 1.0) / (hi - lo)
    w = _smoothstep((u - 0.30) / 0.40)
    cs = (1 - w)[:, None, None] * prof[i0] + w[:, None, None] * prof[(i0 + 1) % len(sch)]
    r_want = np.linalg.norm(cs, axis=2).max(axis=1)          # circumscribed radius

    # ---- shrink where the tube would run into itself ------------------------
    # A tube is embedded if r < 1/kappa everywhere and, for every pair of
    # stations that could be "doubly critical", r_i + r_j < |C_i - C_j|.  No
    # doubly critical pair can exist at arc distance below pi / kappa_max, so
    # closer pairs are covered by the curvature bound alone.
    local_arc = 0.95 * np.pi / kappa.max()
    idx = np.arange(n_a)
    dij = np.minimum(np.abs(idx[:, None] - idx[None, :]),
                     n_a - np.abs(idx[:, None] - idx[None, :])) * ds
    D = np.where(dij >= local_arc,
                 np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2), np.inf)
    r = np.minimum(r_want, CURV_FRAC / np.maximum(kappa, 1e-12))
    for _ in range(80):                                      # monotone relaxation
        f = np.minimum(1.0, ((D - SLOT_GAP) - r[None, :]).min(axis=1) / np.maximum(r, 1e-12))
        if f.min() > 1.0 - 1e-6:
            break
        r = r * np.clip(f, 0.0, 1.0)
    r = np.minimum(r, r_want)
    r = _erode(r, 3)
    r = np.minimum(r, _movavg(r, 7))                         # both only decrease r
    ok = (r[:, None] + r[None, :] + SLOT_GAP <= D + 1e-12)
    assert ok[np.isfinite(D)].all(), "internal error: radius bound violated"

    cs = cs * (r / r_want)[:, None, None]

    # ---- sweep --------------------------------------------------------------
    verts = (pts[:, None, :]
             + cs[:, :, 0:1] * U[:, None, :]
             + cs[:, :, 1:2] * W[:, None, :]).reshape(-1, 3)

    i = np.arange(n_a)[:, None]
    j = np.arange(n_c)[None, :]
    a = i * n_c + j
    b = i * n_c + (j + 1) % n_c
    c = ((i + 1) % n_a) * n_c + (j + 1) % n_c
    d = ((i + 1) % n_a) * n_c + j
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                            np.stack([a, c, d], -1).reshape(-1, 3)])

    unit = 1.0
    if TARGET_SIZE:
        ext = verts.max(0) - verts.min(0)
        unit = TARGET_SIZE / ext.max()
        verts = (verts - 0.5 * (verts.max(0) + verts.min(0))) * unit

    info = dict(L=L * unit, unit=unit, kappa=kappa, delta=delta,
                r_out=r * unit, r_in=np.linalg.norm(cs, axis=2).min(axis=1) * unit,
                clamped=float((r < r_want - 1e-9).mean()), local_arc=local_arc,
                D=D, pts=pts, cs=cs * unit, schedule=sch)
    return verts, faces, info


# ============================== verification =================================


def check_topology(verts, faces):
    out = {}
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, cnt = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    _, dcnt = np.unique(e[:, 0].astype(np.int64) * len(verts) + e[:, 1], return_counts=True)
    out["vertices"] = len(verts)
    out["edges"] = len(cnt)
    out["triangles"] = len(faces)
    out["euler_characteristic"] = len(verts) - len(cnt) + len(faces)
    out["genus"] = 1 - out["euler_characteristic"] // 2
    out["non_manifold_edges"] = int((cnt > 2).sum())
    out["boundary_edges"] = int((cnt == 1).sum())
    out["inconsistently_wound_edges"] = int((dcnt != 1).sum())
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out["degenerate_triangles"] = int((ar <= 0.0).sum())
    out["surface_area"] = float(ar.sum())
    out["signed_volume"] = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    _, cc = np.unique(np.round(verts, 9), axis=0, return_counts=True)
    out["duplicate_vertices"] = int((cc > 1).sum())
    el = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                   np.linalg.norm(a - c, axis=1)], 1)
    out["edge_length_min"] = float(el.min())
    out["edge_length_max"] = float(el.max())
    out["triangle_aspect_max"] = float((el.max(1) / el.min(1)).max())
    return out


def check_cross_sections(cs):
    """Every cross section polygon must be simple, else the tube is pinched."""
    n = cs.shape[1]
    A, B = cs, np.roll(cs, -1, axis=1)
    pi = np.array([i for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)])
    pj = np.array([j for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)])
    p, rr = A[:, pi], B[:, pi] - A[:, pi]
    q, ss = A[:, pj], B[:, pj] - A[:, pj]
    den = rr[..., 0] * ss[..., 1] - rr[..., 1] * ss[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * ss[..., 1] - qp[..., 1] * ss[..., 0]) / den
        v = (qp[..., 0] * rr[..., 1] - qp[..., 1] * rr[..., 0]) / den
    hit = (np.abs(den) > 1e-18) & (t > 1e-9) & (t < 1 - 1e-9) & (v > 1e-9) & (v < 1 - 1e-9)
    return int(hit.any(axis=1).sum())


def _seg_tri(P0, P1, A, B, C):
    E1, E2, Dv = B - A, C - A, P1 - P0
    Pv = np.cross(Dv, E2)
    det = np.einsum("ij,ij->i", E1, Pv)
    ok = np.abs(det) > 1e-24
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    T = P0 - A
    Qv = np.cross(T, E1)
    u = np.einsum("ij,ij->i", T, Pv) * inv
    v = np.einsum("ij,ij->i", Dv, Qv) * inv
    t = np.einsum("ij,ij->i", E2, Qv) * inv
    eps = 1e-12
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def check_self_intersections(verts, faces, block=64):
    """Exact test on every triangle pair whose bounding boxes overlap.

    Pairs are generated and tested one grid cell at a time so peak memory stays
    bounded; a pair straddling several cells may be tested more than once, which
    costs a little time but cannot hide an intersection.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    cell = 2.0 * float(np.percentile(hi - lo, 99.5))
    org = verts.min(0)
    ilo = np.floor((lo - org) / cell).astype(np.int64)
    ihi = np.floor((hi - org) / cell).astype(np.int64)
    grid = {}
    for f in range(len(faces)):
        for x in range(ilo[f, 0], ihi[f, 0] + 1):
            for y in range(ilo[f, 1], ihi[f, 1] + 1):
                for z in range(ilo[f, 2], ihi[f, 2] + 1):
                    grid.setdefault((x, y, z), []).append(f)

    n_tested, n_hit, examples = 0, 0, []
    for bucket in grid.values():
        m = len(bucket)
        if m < 2:
            continue
        v = np.asarray(bucket)
        col = np.arange(m)[None, :]
        for i0 in range(0, m, block):
            row = np.arange(i0, min(i0 + block, m))[:, None]
            sel = col > row
            if not sel.any():
                continue
            pi = v[np.broadcast_to(row, (len(row), m))[sel]]
            pj = v[np.broadcast_to(col, (len(row), m))[sel]]
            k = np.all(lo[pi] <= hi[pj], 1) & np.all(lo[pj] <= hi[pi], 1)
            pi, pj = pi[k], pj[k]
            if not len(pi):
                continue
            fi, fj = faces[pi], faces[pj]
            share = np.zeros(len(pi), bool)
            for x in range(3):
                for y in range(3):
                    share |= fi[:, x] == fj[:, y]   # neighbours may legitimately touch
            pi, pj = pi[~share], pj[~share]
            if not len(pi):
                continue
            n_tested += len(pi)
            hit = np.zeros(len(pi), bool)
            for X, Y in ((pi, pj), (pj, pi)):
                A, B, C = verts[faces[Y, 0]], verts[faces[Y, 1]], verts[faces[Y, 2]]
                for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                    hit |= _seg_tri(verts[faces[X, e0]], verts[faces[X, e1]], A, B, C)
            n_hit += int(hit.sum())
            if hit.any() and len(examples) < 5:
                examples += list(zip(pi[hit][:5].tolist(), pj[hit][:5].tolist()))
    return n_hit, n_tested, examples[:5]


def min_slot_width(verts, window=60):
    """Smallest distance between two surface points on well separated stations."""
    st = np.arange(len(verts)) // N_CROSS
    radius = 0.02 * float((verts.max(0) - verts.min(0)).max())
    org = verts.min(0)
    cell = np.floor((verts - org) / radius).astype(np.int64)
    grid = {}
    for i, k in enumerate(map(tuple, cell)):
        grid.setdefault(k, []).append(i)
    best = np.inf
    offs = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    for k, vv in grid.items():
        cand = []
        for o in offs:
            cand += grid.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), [])
        if not cand:
            continue
        vv, cand = np.array(vv), np.array(cand)
        for c0 in range(0, len(cand), 4096):
            cc = cand[c0:c0 + 4096]
            sd = np.abs(st[vv][:, None] - st[cc][None, :])
            sd = np.minimum(sd, N_ALONG - sd)
            if not (sd > window).any():
                continue
            d = np.linalg.norm(verts[vv][:, None, :] - verts[cc][None, :, :], axis=2)
            best = min(best, float(np.where(sd > window, d, np.inf).min()))
    return best


def report(verts, faces, info):
    print("-" * 74)
    print("geometry")
    print("  bounding box            %.3f x %.3f x %.3f" % tuple(verts.max(0) - verts.min(0)))
    print("  centreline length       %.2f" % info["L"])
    print("  min curvature radius    %.3f" % (info["unit"] / info["kappa"].max()))
    print("  circumscribed radius    %.4f .. %.4f" % (info["r_out"].min(), info["r_out"].max()))
    print("  inscribed radius        %.4f .. %.4f" % (info["r_in"].min(), info["r_in"].max()))
    print("  local feature size span %.0f : 1" % (info["r_out"].max() / info["r_in"].min()))
    print("  thinnest wall           %.4f  (thin shell / ribbon)" % (2 * info["r_in"].min()))
    print("  thinnest rod diameter   %.4f" % (2 * info["r_out"].min()))
    print("  radius clamped on       %.1f%% of stations (near-contact regions)"
          % (100 * info["clamped"]))

    t = check_topology(verts, faces)
    print("topology / watertightness")
    print("  V, E, F                 %d, %d, %d" % (t["vertices"], t["edges"], t["triangles"]))
    print("  Euler characteristic    %d   -> genus %d (closed torus, as designed)"
          % (t["euler_characteristic"], t["genus"]))
    print("  non-manifold edges      %d   <- must be 0" % t["non_manifold_edges"])
    print("  boundary/open edges     %d   <- must be 0" % t["boundary_edges"])
    print("  inconsistent windings   %d   <- must be 0" % t["inconsistently_wound_edges"])
    print("  degenerate triangles    %d   <- must be 0" % t["degenerate_triangles"])
    print("  duplicate vertices      %d   <- must be 0" % t["duplicate_vertices"])
    print("  signed volume           %.3f  (positive -> normals point outward)"
          % t["signed_volume"])
    print("  surface area            %.3f" % t["surface_area"])
    print("mesh quality")
    print("  edge length             %.4f .. %.4f" % (t["edge_length_min"], t["edge_length_max"]))
    print("  worst triangle aspect   %.1f" % t["triangle_aspect_max"])
    bad = check_cross_sections(info["cs"])
    print("  self-crossing sections  %d   <- must be 0" % bad)

    if RUN_SELF_INTERSECTION:
        print("self-intersection (exact triangle-triangle, this takes a moment)")
        n_hit, n_pairs, ex = check_self_intersections(verts, faces)
        print("  candidate pairs tested  %d" % n_pairs)
        print("  intersecting pairs      %d   <- must be 0" % n_hit)
        if ex:
            print("  example bad face pairs  %s" % ex)
        g = min_slot_width(verts)
        print("  narrowest slot          %.4f  (distinct strands nearly touching)" % g)
    print("-" * 74)


def check_contacts(info):
    """Warn if CONTACTS no longer matches the knot the user configured."""
    cl = info["D"].min(axis=1)
    n = len(cl)
    mins = [j for j in range(n) if cl[j] <= cl[(j - 1) % n] and cl[j] <= cl[(j + 1) % n]
            and cl[j] < 1.02 * cl.min()]
    fr = sorted({round(j / n, 3) for j in mins})
    if any(min(abs(np.array(fr) - c)) > 0.02 for c in CONTACTS) and fr:
        print("NOTE: closest-approach arc fractions for this knot are approximately")
        print("      %s" % fr[:8])
        print("      set CONTACTS to four of these to keep the narrow slots.")


# ============================== exporters ====================================


def write_ply(path, verts, faces):
    v = np.asarray(verts, "<f4")
    f = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
    f["n"] = 3
    f["i"] = faces
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n"
                  "element vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                  "element face %d\nproperty list uchar int vertex_indices\n"
                  "end_header\n" % (len(v), len(f))).encode())
        fh.write(v.tobytes())
        fh.write(f.tobytes())


def write_stl(path, verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = np.stack([a, b, c], 1)
    with open(path, "wb") as fh:
        fh.write(b"hard_geometry_knot".ljust(80, b"\0"))
        fh.write(np.array([len(faces)], "<u4").tobytes())
        fh.write(rec.tobytes())


def write_obj(path, verts, faces):
    with open(path, "w") as fh:
        fh.write("# hard_geometry_knot\n")
        np.savetxt(fh, verts, fmt="v %.9g %.9g %.9g")
        np.savetxt(fh, faces + 1, fmt="f %d %d %d")


# ================================ blender ====================================


def to_blender(verts, faces, name):
    import bpy
    import bmesh
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in verts], [], [tuple(t) for t in faces])
    nv, nf = len(me.vertices), len(me.polygons)
    if me.validate(verbose=False):
        print("WARNING: Blender's mesh validation altered the mesh "
              "(%d -> %d verts, %d -> %d faces)" % (nv, len(me.vertices), nf, len(me.polygons)))
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

    bm = bmesh.new()
    bm.from_mesh(me)
    print("blender side check")
    print("  verts / faces           %d / %d" % (len(bm.verts), len(bm.faces)))
    print("  non-manifold edges      %d" % sum(1 for e in bm.edges if not e.is_manifold))
    print("  boundary edges          %d" % sum(1 for e in bm.edges if e.is_boundary))
    print("  loose verts             %d" % sum(1 for v in bm.verts if not v.link_faces))
    print("  bmesh volume            %.4f" % bm.calc_volume(signed=True))
    bm.free()
    return ob


# ================================= main ======================================


def main():
    print("building %d triangles ..." % (2 * N_ALONG * N_CROSS))
    verts, faces, info = build()
    for p, label, _ in sorted(info["schedule"], key=lambda z: z[0]):
        print("  s=%.3f  %s" % (p, label))
    if RUN_CHECKS:
        report(verts, faces, info)
        check_contacts(info)

    try:
        import bpy  # noqa: F401
        to_blender(verts, faces, OBJECT_NAME)
        print("created object '%s' in the scene" % OBJECT_NAME)
    except ImportError:
        print("no bpy available, skipping scene creation")

    if EXPORT_DIR:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        for ext, fn in (("ply", write_ply), ("stl", write_stl), ("obj", write_obj)):
            p = os.path.join(EXPORT_DIR, OBJECT_NAME + "." + ext)
            fn(p, verts, faces)
            print("wrote %s  (%.1f MB)" % (p, os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
