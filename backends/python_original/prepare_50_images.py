#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import rasterio as rio
import tacoreader.v1 as tacoreader
from rasterio.windows import Window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="benchmark/data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
    window = Window(0, 0, 512, 512)

    made = 0
    idx = args.start
    while made < args.count:
        try:
            sample = dataset.read(idx)
            with rio.open(sample.read(0)) as src:
                image = src.read(window=window).astype(np.float32) / 10000.0
            if image.shape != (13, 512, 512):
                raise ValueError(f"unexpected shape {image.shape}")
            path = out / f"sample_{made:03d}_dataset_{idx:05d}.bin"
            image[np.newaxis].astype(np.float32).tofile(path)
            print(path)
            made += 1
        except Exception as exc:
            print(f"skip dataset index {idx}: {exc}")
        idx += 1

    print(f"Prepared {made} real 13-band images in {out}")


if __name__ == "__main__":
    main()
