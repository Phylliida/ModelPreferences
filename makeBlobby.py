import numpy as np
import shapely.geometry as geom
import shapely.ops as ops
import alphashape


def blobby(points,
           alpha=None,          # how “tight” the concave hull is
           smooth_px=3,         # how many pixels of smoothing
           oversample=4):       # density of evaluated boundary
    """
    Return a Shapely Polygon whose outline is a smooth blob
    surrounding all input 2-D points.

    Parameters
    ----------
    points : (n,2) array-like
    alpha  : float or None
        α parameter for the α-shape.  If None we let
        alphashape.optimizealpha() choose.
    smooth_px : positive float
        Roughly how far (in data units) the boundary is allowed
        to wiggle while being smoothed.  Larger == softer edges.
    oversample : int
        Controls the number of vertices put on the final outline.
    """

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be of shape (k,2)")

    # ------------------------------------------------------------------
    # 1) Concave hull (α-shape)
    # ------------------------------------------------------------------
    if alpha is None:
        alpha = alphashape.optimizealpha(pts)
    hull = alphashape.alphashape(pts, alpha)

    # ------------------------------------------------------------------
    # 2) Smoothing.  Two passes of “buffer-in buffer-out” is
    #    Chaikin-like corner rounding.  We keep shrinking radius
    #    so we don’t grow the area too much.
    # ------------------------------------------------------------------
    radius = smooth_px
    for _ in range(2):
        hull = hull.buffer(+radius,  join_style=1, cap_style=1)
        hull = hull.buffer(-radius,  join_style=1, cap_style=1)
        radius *= 0.5                           # progressively finer

    # ------------------------------------------------------------------
    # 3) Optional vertex densification so the outline looks smooth
    #    when plotted.  We interpolate points every 1/oversample *
    #    average edge length.
    # ------------------------------------------------------------------
    hull = ops.transform(
        lambda x, y, z=None:
        _densify(x, y, oversample),
        hull
    )

    return hull


# ----------------------------------------------------------------------
# Helper: densify LineString/Polygon without changing its shape
# ----------------------------------------------------------------------
def _densify(xs, ys, over):
    import numpy as np
    xy = np.column_stack([xs, ys])
    # drop last point (duplicate of first for polygons)
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
        closed = True
    else:
        closed = False

    # cumulative chord length
    seglen = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    cum = np.insert(seglen.cumsum(), 0, 0.0)
    total = cum[-1]
    step = total / (len(xy) * over)
    new_s = np.arange(0, total, step)

    # linear interpolation
    new_xy = np.vstack([
        np.interp(new_s, cum, xy[:, 0]),
        np.interp(new_s, cum, xy[:, 1])
    ]).T

    if closed:
        new_xy = np.vstack([new_xy, new_xy[0]])

    return new_xy[:, 0], new_xy[:, 1]

import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import mapping

# Random point cloud
np.random.seed(0)
pts = np.random.normal(size=(50, 2)) * [1.5, 0.7] + [0, 1]

blob = blobby(pts, smooth_px=0.3)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(pts[:, 0], pts[:, 1], s=20, color="k", zorder=3)

# Plot the blob
x, y = blob.exterior.xy
ax.fill(x, y, color="#1f77b4", alpha=.35, zorder=2)
ax.plot(x, y, color="#1f77b4", lw=2)

ax.set_aspect("equal")
ax.set_title("Smooth blob around k points")
plt.show()