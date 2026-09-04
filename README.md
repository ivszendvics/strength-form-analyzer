# Strength Form Analyzer

A computer-vision system for analyzing strength-training movements using pose estimation, biomechanical joint angles, temporal analysis, and rule-based form evaluation.

Give it a video of a squat, deadlift, lunge, or bench press and it will detect the person's pose frame-by-frame, compute joint angles, automatically segment the video into repetitions, score each rep against a set of explainable, configurable heuristics, and produce an annotated video, summary plots, and machine-readable JSON/CSV results.

> **This is a form-analysis heuristics tool, not a medical or coaching product.** See [Limitations](#limitations) before drawing any conclusions from its output.

## Features

- **Pose estimation** via MediaPipe's Pose Landmarker, behind a backend-agnostic interface so another estimator (e.g. OpenPose) could be added later.
- **Geometric joint-angle calculations** (knee, hip, trunk) with confidence-aware handling of missing/low-visibility landmarks.
- **Signal smoothing** (moving average, exponential, Savitzky-Golay) to reduce frame-to-frame jitter without destroying true peaks/valleys.
- **Automatic rep detection** via a hysteresis-based state machine (not naive local-extrema detection).
- **Per-rep metrics**: duration, tempo, range of motion, min/max angles, descent/ascent split, left/right asymmetry.
- **Rule-based form analysis** producing a traceable, weighted score per rep (depth, trunk lean, tempo, consistency, symmetry, ...), each with named, configurable thresholds.
- **GOOD / NEEDS_IMPROVEMENT / UNCERTAIN** classification -- the system says "I don't know" rather than guessing when pose confidence is too low.
- **Visualization**: an annotated output video (skeleton, live angles, rep/phase, form score, warnings) plus summary plots (angle-vs-time with rep boundaries, form score per rep, tempo per rep, ROM per rep).
- **JSON/CSV export** of all per-rep results.
- **A CLI** with configurable smoothing, confidence threshold, frame-step (speed/resolution tradeoff), and per-output toggles.

## Demo

_A demo GIF/video showing the annotated output goes here once one is recorded. Run the CLI on your own video (see [Usage](#usage)) and drop `outputs/annotated.mp4` (or a short GIF made from it) into this section._

## Supported Exercises

| Exercise | Tracked joints | Computed angles | Rep detection driven by |
|---|---|---|---|
| Squat | hip, knee, ankle, shoulder | knee, hip, trunk | knee angle |
| Deadlift | shoulder, hip, knee, ankle | hip, knee, trunk | hip angle |
| Lunge | hip, knee, ankle | front knee, rear knee, hip, trunk | front knee angle |
| Bench Press | shoulder, elbow, wrist, hip | elbow, shoulder (elbow flare) | elbow angle |

Adding a new exercise means implementing the `Exercise` interface (`src/exercises/base.py`) and adding a `configs/<name>.yaml` -- no changes to pose estimation, rep detection, scoring, visualization, or the CLI are needed. See `src/exercises/squat.py` for a complete example.

## Architecture

```
strength-form-analyzer/
├── src/
│   ├── main.py              # CLI entry point
│   ├── pose/                # Pose estimation backend abstraction + MediaPipe implementation
│   ├── biomechanics/        # Angle geometry, smoothing, per-rep metric extraction
│   ├── exercises/           # Exercise plug-ins: squat, deadlift, lunge, bench press
│   ├── reps/                # Rep-detection state machine
│   ├── analysis/            # Form rules, weighted scoring, orchestration
│   ├── visualization/       # Video overlay + summary plots
│   └── io/                  # JSON/CSV export
├── tests/                   # Unit + synthetic-data integration tests (no video/GPU required)
├── configs/                 # Per-exercise thresholds and scoring weights (YAML)
├── data/                    # Local video input (gitignored)
├── outputs/                 # Generated results (gitignored)
└── notebooks/                # Exploratory analysis demo
```

Every stage of the pipeline depends only on the abstraction of the stage before it (pose estimator interface, named angle dict, generic `Rep`/`RepMetrics`, generic `RuleResult`), which is what makes each stage independently testable and swappable.

## Installation

Requires Python 3.10-3.12 (MediaPipe does not yet publish wheels for newer versions).

```bash
git clone <this-repo-url>
cd strength-form-analyzer
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a fully reproducible install (the exact versions this project was last tested against, rather than the widest compatible range), use `pip install -r requirements-lock.txt` instead. See the comment at the top of that file for how it's generated/regenerated.

The first run downloads a MediaPipe pose model bundle (a few MB) to `~/.cache/strength_form_analyzer/models/` -- this is machine-local and never part of the repo.

## Usage

```bash
python -m src.main --video data/input/squat.mp4 --exercise squat --output outputs/
```

```bash
python -m src.main --help
```

Useful options:

```text
--video               Path to the input video file (required)
--exercise            squat | deadlift | lunge (required)
--output              Output directory (required)
--config              Override the exercise's YAML config path
--confidence-threshold  Minimum landmark visibility to trust (default: 0.5)
--smoothing           moving_average | exponential | savgol (default: savgol)
--frame-step          Process every Nth frame (default: 1)
--model-complexity    lite | full | heavy MediaPipe model (default: lite)
--save-video/--no-save-video
--save-plots/--no-save-plots
--save-json/--no-save-json
--save-csv/--no-save-csv
--verbose
```

Example: faster, lower-resolution-in-time processing on a long video, video output skipped:

```bash
python -m src.main --video data/input/deadlift_set.mp4 --exercise deadlift \
    --output outputs/deadlift_set --frame-step 2 --no-save-video
```

## Output

Running the CLI produces, inside `--output`:

```text
outputs/
├── results.json
├── results.csv
├── annotated.mp4       (unless --no-save-video)
└── plots/              (unless --no-save-plots)
    ├── knee_angle_over_time.png
    ├── hip_angle_over_time.png
    ├── trunk_angle_over_time.png
    ├── form_score_by_rep.png
    ├── rep_duration_by_rep.png
    └── range_of_motion_by_rep.png
```

Example per-rep entry from `results.json`:

```json
{
  "rep_number": 4,
  "start_frame": 118,
  "bottom_frame": 142,
  "end_frame": 171,
  "duration_seconds": 1.77,
  "metrics": {
    "duration_seconds": 1.77,
    "descent_duration_seconds": 0.8,
    "ascent_duration_seconds": 0.97,
    "tempo_ratio": 0.825,
    "range_of_motion_degrees": { "knee_angle": 82.3 },
    "min_angle_degrees": { "knee_angle": 91.2, "trunk_angle": 2.1 },
    "max_angle_degrees": { "trunk_angle": 38.4 }
  },
  "form": {
    "rep": 4,
    "classification": "NEEDS_IMPROVEMENT",
    "overall_score": 82.4,
    "components": [
      { "name": "depth", "score": 100.0, "weight": 0.3 },
      { "name": "trunk", "score": 61.0, "weight": 0.2 },
      { "name": "tempo", "score": 94.0, "weight": 0.15 },
      { "name": "stability", "score": 100.0, "weight": 0.2 },
      { "name": "symmetry", "score": 100.0, "weight": 0.1 },
      { "name": "knee_tracking", "score": 100.0, "weight": 0.05 }
    ],
    "issues": [
      {
        "type": "EXCESSIVE_TRUNK_LEAN",
        "severity": "medium",
        "confidence": 0.71,
        "message": "Potential issue: peak trunk inclination of 38 degrees from vertical exceeds the configured comfort threshold of 45 degrees."
      }
    ]
  }
}
```

Every classification is traceable to its named, weighted components -- e.g.:

```text
Rep 4
 ├── Depth: 100/100  (weight 0.30)
 ├── Trunk: 61/100   (weight 0.20)  -> EXCESSIVE_TRUNK_LEAN (medium, conf 0.71)
 ├── Tempo: 94/100   (weight 0.15)
 ├── Stability: 100/100 (weight 0.20)
 ├── Symmetry: 100/100  (weight 0.10)
 ├── Knee tracking: 100/100 (weight 0.05)
 └── Overall: 82.4/100 -> NEEDS_IMPROVEMENT (< 80 threshold... this example
     is actually at the boundary; see configs/squat.yaml `good_threshold`)
```

## Methodology

```text
Video
 ↓
Pose Estimation            (src/pose)
 ↓
Landmark Tracking          (src/pose/landmarks.py)
 ↓
Joint Angle Calculation    (src/biomechanics/angles.py)
 ↓
Signal Smoothing           (src/biomechanics/smoothing.py)
 ↓
Rep Detection              (src/reps/detector.py)
 ↓
Feature Extraction         (src/biomechanics/metrics.py)
 ↓
Form Analysis              (src/analysis/rules.py, src/exercises)
 ↓
Rep Classification         (src/analysis/scoring.py)
 ↓
Visualization               (src/visualization)
```

### Rep detection

Rep detection is a state machine (`IDLE -> DESCENDING -> ASCENDING -> COMPLETED -> IDLE`) driven by a single exercise-specific "primary" angle crossing configurable, hysteresis-separated thresholds -- e.g. for a squat, the knee angle must drop meaningfully below a "standing" threshold to start a rep, reach a genuine minimum depth, and climb back to near-standing before the rep is accepted. The bottom frame isn't a separate dwell state (which frame was the true bottom is only knowable in hindsight); it's tracked as a running minimum while descending/ascending. This -- plus a minimum rep duration and a consecutive-frame debounce requirement -- is what prevents small noisy fluctuations from registering as fake reps, which naive local-minimum/maximum detection would produce.

### Form scoring

Each rep is scored by several independent, named rule functions (`src/analysis/rules.py`) -- e.g. depth, trunk lean, tempo consistency, rep-to-rep consistency, left/right symmetry -- each producing a 0-100 component score plus zero or more specific issues with a severity and confidence. These are combined as a **configurable weighted average** (`src/analysis/scoring.py`) into one overall score and a `GOOD` / `NEEDS_IMPROVEMENT` / `UNCERTAIN` classification. There is no single hardcoded `if angle < X` rule anywhere in the classification path -- every classification can be traced back to which named components contributed what, and why (see the JSON `form.components` / `form.issues` breakdown above).

If pose confidence during a rep was too low (occlusion, motion blur, the person leaving frame), the rep is classified `UNCERTAIN` regardless of its computed score, rather than confidently guessing.

## Camera Assumptions

- **Squat / Deadlift**: side or front-side camera, full body visible, stable camera, adequate lighting, minimal occlusion.
- **Lunge**: side view of a forward/reverse split stance, both legs visible.
- **Bench Press**: side view of the bench, the pressing arm (shoulder through wrist) and torso fully visible throughout the full range of motion.
- A single person in frame. Multiple people are not reliably supported (see Limitations).
- Angles are computed from whichever side (left/right) has higher landmark visibility per frame, which degrades gracefully for a side-on camera without assuming a fixed camera side -- but this is still a **2D projection** of a 3D movement (see below).

## Limitations

- **2D pose estimation**: joint angles are computed from a single camera's 2D projection of 3D movement. The measured angle can differ meaningfully from the true 3D joint angle depending on how the movement plane aligns with the camera.
- **Camera angle**: results assume the documented camera setup above; an off-axis or handheld camera will distort measured angles.
- **Occlusion**: a limb hidden behind the body, other equipment, or clothing will produce missing/low-confidence landmarks; the pipeline marks affected reps `UNCERTAIN` rather than guessing, but can't recover the missing data.
- **Lighting**: poor lighting reduces MediaPipe's landmark confidence.
- **Loose clothing**: baggy clothing obscures the true joint location MediaPipe is trying to estimate.
- **Multiple people**: only the most prominent detected person is used; a second person in frame is not filtered out and can degrade results.
- **Individual anatomical differences**: "ideal" depth/angle thresholds vary by limb-length ratios, hip structure, and ankle/shoulder mobility -- the shipped thresholds are reasonable starting points, not universal truths.
- **Exercise-specific variation**: technique varies by training style (e.g. high-bar vs. low-bar squat, conventional vs. sumo deadlift) in ways the current rules don't distinguish.
- **Heuristic thresholds**: every threshold in `configs/*.yaml` is a documented starting-point heuristic. None of them were fit to labeled real-world data -- see [Data and Calibration](#data-and-calibration).
- **Not medical or coaching advice**: this tool describes results as form-analysis heuristics. It cannot and does not make a medically or professionally authoritative determination about injury risk or technique correctness.

## Data and Calibration

The initial version is intentionally a **transparent, rules-based system**, not a machine-learning classifier -- the goal is to demonstrate geometry, time-series/signal-processing, and feature-engineering fundamentals with fully explainable output. The architecture (named component scores feeding a weighted combiner) is deliberately structured so that a future version could replace individual rule functions, or the weight-combination step itself, with a model trained on labeled reps, without changing the rest of the pipeline. Before relying on the shipped thresholds for anything beyond a demo/portfolio project, calibrate `configs/*.yaml` against your own labeled examples of good/bad reps.

## Future Work

- OpenPose (or another) backend behind the existing `PoseEstimator` interface
- 3D pose estimation to reduce single-camera projection error
- Object detection for barbell tracking / bar-path visualization
- A learned form classifier trained on labeled reps, replacing or augmenting the rule-based scorer
- Per-user threshold calibration from a short labeled warm-up set
- Multi-camera analysis (front + side simultaneously)
- Real-time webcam mode
- A web interface
- Automatic exercise-type detection instead of requiring `--exercise`

## Testing

```bash
pip install -r requirements.txt
pytest
```

All tests run against synthetic angle sequences and geometric fixtures -- no video file or GPU is required.

## License

[MIT](LICENSE)
