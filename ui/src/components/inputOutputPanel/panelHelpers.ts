export const ASPECT_RATIOS = [
  { label: "FREE", value: null },
  { label: "16:9", value: 16 / 9 },
  { label: "9:16", value: 9 / 16 },
  { label: "4:5", value: 0.8 },
  { label: "1:1", value: 1 },
  { label: "2.35:1", value: 2.35 },
];

export const FPS_OPTIONS = [
  { value: 0, label: "NATIVE", sub: "SOURCE" },
  { value: 30, label: "30 FPS", sub: "STD" },
  { value: 60, label: "60 FPS", sub: "SMOOTH" },
  { value: 120, label: "120 FPS", sub: "SLOW-MO" },
];

export function truncateModelName(id: string, maxLen = 12): string {
  if (id.length <= maxLen) return id;
  let short = id
    .replace("RealESRGAN_", "ESRGAN-")
    .replace("_x4plus", "")
    .replace("_anime_6B", "-ANI");
  if (short.length <= maxLen) return short;
  return `${short.slice(0, maxLen - 1)}\u2026`;
}

export function getSmartResInfo(w: number, h: number) {
  if (w === 0 || h === 0) return { label: "---", detail: "" };
  const min = Math.min(w, h);
  const dims = `${w} × ${h}`;
  if (min >= 4320) return { label: "8K UHD", detail: dims };
  if (min >= 2160) return { label: "4K UHD", detail: dims };
  if (min >= 1440) return { label: "1440p QHD", detail: dims };
  if (min >= 1080) return { label: "1080p FHD", detail: dims };
  if (min >= 720) return { label: "720p HD", detail: dims };
  if (min >= 480) return { label: "480p SD", detail: dims };
  return { label: dims, detail: "" };
}
