from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request
import os


TEAPOT_URL = "https://github.com/jaz303/utah-teapot/raw/refs/heads/master/teapot.obj"

# Make matplotlib/fontconfig caches writable in sandboxes.
_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_ROOT / ".cache"))


@dataclass(frozen=True)
class Settings:
    H: int = 96
    W: int = 96
    fx: float = 120.0
    fy: float = 120.0
    z_offset: float = 3.0
    light_pos: tuple[float, float, float] = (2.0, 2.0, 1.0)
    light_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    num_indirect_samples: int = 6


def download_teapot(dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    print(f"Downloading teapot to {dst_path} ...")
    urllib.request.urlretrieve(TEAPOT_URL, dst_path)


def normalize_vertices_torch(V, z_offset: float):
    # Center
    vmin = V.min(dim=0).values
    vmax = V.max(dim=0).values
    center = (vmin + vmax) / 2.0
    V = V - center

    # Scale
    scale = V.abs().max()
    V = V / (scale + 1e-8)

    # Move in front of camera (+z)
    V[:, 2] = V[:, 2] + z_offset
    return V


def make_rays_pinhole_torch(H, W, fx, fy, cam2world, cx=None, cy=None):
    device = cam2world.device
    dtype = cam2world.dtype

    if cx is None:
        cx = (W - 1) / 2.0
    if cy is None:
        cy = (H - 1) / 2.0

    u = cam2world.new_tensor(range(W), dtype=dtype) + 0.5
    v = cam2world.new_tensor(range(H), dtype=dtype) + 0.5
    vv, uu = __import__("torch").meshgrid(v, u, indexing="ij")

    x = (uu - cx) / fx
    y = -(vv - cy) / fy
    z = __import__("torch").ones_like(x)

    d_cam = __import__("torch").stack([x, y, z], dim=-1)
    d_cam = d_cam / (__import__("torch").linalg.norm(d_cam, dim=-1, keepdim=True) + 1e-8)

    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    ray_d = d_cam @ R.T
    ray_d = ray_d / (__import__("torch").linalg.norm(ray_d, dim=-1, keepdim=True) + 1e-8)
    ray_o = t.view(1, 1, 3).expand_as(ray_d).clone()
    return ray_o, ray_d


def intersect_rays_triangles_torch(ray_o, ray_d, V, F, eps=1e-8):
    torch = __import__("torch")
    device = ray_o.device
    dtype = ray_o.dtype

    H, W, _ = ray_o.shape
    M = F.shape[0]

    depth = torch.full((H, W), float("inf"), device=device, dtype=dtype)
    tri_id = torch.full((H, W), -1, device=device, dtype=torch.int64)

    Rn = H * W
    ro = ray_o.reshape(Rn, 3)
    rd = ray_d.reshape(Rn, 3)

    for i in range(M):
        i0, i1, i2 = F[i]
        v0 = V[i0]
        v1 = V[i1]
        v2 = V[i2]

        e1 = v1 - v0
        e2 = v2 - v0

        pvec = torch.cross(rd, e2.expand_as(rd), dim=-1)
        det = (e1 * pvec).sum(dim=-1)

        valid_det = det.abs() > eps
        inv_det = torch.where(valid_det, 1.0 / det, torch.zeros_like(det))

        tvec = ro - v0
        u = (tvec * pvec).sum(dim=-1) * inv_det

        qvec = torch.cross(tvec, e1.expand_as(tvec), dim=-1)
        v = (rd * qvec).sum(dim=-1) * inv_det

        t = (e2 * qvec).sum(dim=-1) * inv_det

        hit_i = valid_det & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)

        t_img = t.reshape(H, W)
        closer = hit_i.reshape(H, W) & (t_img < depth)
        depth = torch.where(closer, t_img, depth)
        tri_id = torch.where(closer, torch.tensor(i, device=device, dtype=torch.int64), tri_id)

    hit = tri_id >= 0
    return hit, depth, tri_id


def triangle_normals_torch(V, F, eps=1e-8):
    torch = __import__("torch")
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = torch.cross(e1, e2, dim=-1)
    n = n / (torch.linalg.norm(n, dim=-1, keepdim=True) + eps)
    return n


def build_normal_map_torch(N_tri, tri_id):
    torch = __import__("torch")
    H, W = tri_id.shape
    normal_map = torch.zeros((H, W, 3), device=tri_id.device, dtype=N_tri.dtype)
    hit = tri_id >= 0
    if hit.any():
        normal_map[hit] = N_tri[tri_id[hit]]
    return normal_map


def lambertian_torch(normal_map, hit_points, light_pos, eps=1e-8):
    torch = __import__("torch")
    light_pos = torch.tensor(light_pos, device=hit_points.device, dtype=hit_points.dtype)
    l = light_pos.view(1, 1, 3) - hit_points
    l = l / (torch.linalg.norm(l, dim=-1, keepdim=True) + eps)
    ndotl = (normal_map * l).sum(dim=-1).clamp(min=0.0)
    return ndotl


def hard_shadows_torch(hit, hit_points, normal_map, light_pos, V, F, eps=1e-4):
    torch = __import__("torch")
    light_pos_t = torch.tensor(light_pos, device=hit_points.device, dtype=hit_points.dtype)
    o = hit_points + eps * normal_map
    toL = light_pos_t.view(1, 1, 3) - o
    dist = torch.linalg.norm(toL, dim=-1)
    d = toL / (dist[..., None] + 1e-8)
    sh_hit, sh_depth, _ = intersect_rays_triangles_torch(o, d, V, F)
    shadow = hit & sh_hit & (sh_depth < dist)
    return shadow


def shade_direct_torch(hit, ray_o, ray_d, depth, normal_map, light_pos, light_rgb, V, F):
    torch = __import__("torch")
    light_rgb_t = torch.tensor(light_rgb, device=depth.device, dtype=depth.dtype)
    hit_points = ray_o + depth[..., None] * ray_d
    shadow = hard_shadows_torch(hit, hit_points, normal_map, light_pos, V, F)
    diffuse = lambertian_torch(normal_map, hit_points, light_pos)
    vis = (~shadow).to(diffuse.dtype)
    intensity = diffuse * vis
    rgb = torch.zeros((*hit.shape, 3), device=depth.device, dtype=depth.dtype)
    rgb[hit] = intensity[hit][..., None] * light_rgb_t.view(1, 1, 3)
    return rgb, shadow


def sample_hemisphere_torch(n, num_samples, eps=1e-8):
    torch = __import__("torch")
    H, W, _ = n.shape
    d = torch.randn((num_samples, H, W, 3), device=n.device, dtype=n.dtype)
    d = d / (torch.linalg.norm(d, dim=-1, keepdim=True) + eps)
    ndot = (d * n.unsqueeze(0)).sum(dim=-1, keepdim=True)
    d = torch.where(ndot >= 0, d, -d)
    return d


def one_bounce_indirect_torch(ray_o, ray_d, V, F, N_tri, light_pos, light_rgb, num_samples=6, eps=1e-4):
    torch = __import__("torch")
    device = ray_o.device
    dtype = ray_o.dtype

    hit1, depth1, tri1 = intersect_rays_triangles_torch(ray_o, ray_d, V, F)
    H, W = hit1.shape
    rgb = torch.zeros((H, W, 3), device=device, dtype=dtype)
    if not hit1.any():
        return rgb

    p1 = ray_o + depth1[..., None] * ray_d
    n1 = torch.zeros((H, W, 3), device=device, dtype=dtype)
    n1[hit1] = N_tri[tri1[hit1]]

    dirs = sample_hemisphere_torch(n1, num_samples=num_samples)
    o2 = (p1 + eps * n1).unsqueeze(0).expand_as(dirs)

    light_pos_t = torch.tensor(light_pos, device=device, dtype=dtype)
    light_rgb_t = torch.tensor(light_rgb, device=device, dtype=dtype)

    acc = torch.zeros((H, W, 3), device=device, dtype=dtype)
    count = torch.zeros((H, W, 1), device=device, dtype=dtype)

    for s in range(num_samples):
        hit2, depth2, tri2 = intersect_rays_triangles_torch(o2[s], dirs[s], V, F)
        if not hit2.any():
            continue

        p2 = o2[s] + depth2[..., None] * dirs[s]
        n2 = torch.zeros((H, W, 3), device=device, dtype=dtype)
        n2[hit2] = N_tri[tri2[hit2]]

        toL = light_pos_t.view(1, 1, 3) - p2
        toL = toL / (torch.linalg.norm(toL, dim=-1, keepdim=True) + 1e-8)
        ndotl = (n2 * toL).sum(dim=-1).clamp(min=0.0)
        contrib = ndotl[..., None] * light_rgb_t.view(1, 1, 3)

        acc[hit2] += contrib[hit2]
        count[hit2] += 1.0

    count = torch.clamp(count, min=1.0)
    rgb[hit1] = (acc / count)[hit1]
    return rgb


def save_rgb(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    import numpy as np

    arr = img.clamp(0.0, 1.0).detach().cpu().numpy()
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def save_scalar_gray(path: Path, x):
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0,1] for visualization
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)

    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def save_normal_rgb(path: Path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    import numpy as np

    viz = (n + 1.0) * 0.5
    viz = viz.clamp(0.0, 1.0).detach().cpu().numpy()
    arr8 = (viz * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def save_mesh_views(path: Path, V):
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    v = V.detach().cpu().numpy()

    def raster(points_xy, size=320):
        # points_xy: (N,2) roughly in [-1,1] range (after normalization)
        img = np.zeros((size, size), dtype=np.uint8)
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        # Normalize to [0, size-1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        if xmax > xmin:
            x = (x - xmin) / (xmax - xmin)
        else:
            x = x * 0
        if ymax > ymin:
            y = (y - ymin) / (ymax - ymin)
        else:
            y = y * 0
        xi = np.clip((x * (size - 1)).astype(np.int32), 0, size - 1)
        yi = np.clip(((1.0 - y) * (size - 1)).astype(np.int32), 0, size - 1)
        img[yi, xi] = 255
        return img

    xy = raster(v[:, [0, 1]])
    xz = raster(v[:, [0, 2]])
    yz = raster(v[:, [1, 2]])

    gap = 8
    H = xy.shape[0]
    canvas = np.zeros((H, H * 3 + gap * 2), dtype=np.uint8)
    canvas[:, 0:H] = xy
    canvas[:, H + gap : 2 * H + gap] = xz
    canvas[:, 2 * H + 2 * gap : 3 * H + 2 * gap] = yz
    Image.fromarray(canvas).save(path)


def main() -> int:
    torch = __import__("torch")
    import trimesh

    root = Path(__file__).resolve().parents[2]
    # Ensure cache dirs exist before importing matplotlib anywhere.
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    assets_img = root / "assets" / "img" / "hands-on-rendering"
    assets_mesh = root / "assets" / "meshes"

    s = Settings()
    teapot_path = assets_mesh / "teapot.obj"
    download_teapot(teapot_path)

    mesh = trimesh.load(teapot_path, force="mesh")
    V = torch.tensor(mesh.vertices, dtype=torch.float32)
    F = torch.tensor(mesh.faces, dtype=torch.int64)
    V = normalize_vertices_torch(V, z_offset=s.z_offset)

    print("Teapot:", {"V": tuple(V.shape), "F": tuple(F.shape)})
    save_mesh_views(assets_img / "lab0_teapot_views.png", V)

    cam2world = torch.eye(4, dtype=torch.float32)
    ray_o, ray_d = make_rays_pinhole_torch(s.H, s.W, s.fx, s.fy, cam2world)

    # Lab 1 viz: ray directions as RGB
    save_normal_rgb(assets_img / "lab1_rays_rgb.png", ray_d)

    # Lab 2: depth + hit
    hit, depth, tri_id = intersect_rays_triangles_torch(ray_o, ray_d, V, F)
    depth_vis = depth.clone()
    if hit.any():
        depth_vis[~hit] = depth[hit].max()
    else:
        depth_vis[:] = 0.0
    save_scalar_gray(assets_img / "lab2_depth.png", depth_vis)

    # Lab 3: normals
    N_tri = triangle_normals_torch(V, F)
    normal_map = build_normal_map_torch(N_tri, tri_id)
    save_normal_rgb(assets_img / "lab3_normals.png", normal_map)

    # Lab 4: direct lighting
    rgb_direct, shadow = shade_direct_torch(hit, ray_o, ray_d, depth, normal_map, s.light_pos, s.light_rgb, V, F)
    save_rgb(assets_img / "lab4_direct.png", rgb_direct)
    save_scalar_gray(assets_img / "lab4_shadow.png", shadow.to(depth.dtype))

    # Lab 5: 1-bounce indirect (indirect-only visualization)
    rgb_indirect = one_bounce_indirect_torch(
        ray_o, ray_d, V, F, N_tri, s.light_pos, s.light_rgb, num_samples=s.num_indirect_samples
    )
    save_rgb(assets_img / "lab5_indirect.png", rgb_indirect)

    print(f"Saved images to: {assets_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

