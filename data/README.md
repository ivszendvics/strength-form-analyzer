# data/

`data/input/` is where you place your own local video files to analyze:

```bash
python -m src.main --video data/input/squat.mp4 --exercise squat --output outputs/
```

This directory (and the video files inside it) is **gitignored** -- videos of
people are personal data and should not be committed to a public repository.
Nothing under `data/input/` is part of this repo except this README.

## Camera setup

See the README's ["Camera Assumptions"](../README.md#camera-assumptions)
section for the recommended framing per exercise (side vs. front-side view,
full body visible, stable camera, adequate lighting, minimal occlusion,
single person in frame).

## Sample videos

No sample videos are bundled with this repository (to keep it small and to
avoid shipping anyone's likeness). If you'd like to try the pipeline without
recording your own footage, use any video you have rights to that shows one
person performing a squat, deadlift, or lunge from a side-on angle.
