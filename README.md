# AI Football Referee

Sick of refs getting it wrong?

Introducing Multi-View Foul Recognition, the AI-powered solution designed to make football refereeing more accurate and fair. By leveraging multiple camera angles, this system can detect fouls with precision and assist referees in making real-time decisions based on comprehensive video analysis. 

<div align="center">
  <img src="./assets/demo1.png" alt="AI Football Referee Interface" width="100%" />
</div>

## 🌟 Overview

The AI Football Referee is a web application that analyzes video clips of football incidents to provide referee decisions with explanations. Deep learning models and computer vision techniques recognize actions like tackles, challenges, and fouls, then produce a referee decision based on official football regulations.

## ✨ Key Features

- **Video Upload & Analysis**: Upload video clips of football incidents for automated analysis
- **Multi-task Deep Learning**: A single model predicts several aspects of an incident at once:
  - Action classification (tackling, holding, elbowing, etc.)
  - Body part involvement
  - Touch ball detection
  - Intent analysis (try to play detection)
  - Severity assessment
- **Referee Decision Generation**: A decision plus a natural-language explanation, from Azure OpenAI when configured and from deterministic rules otherwise
- **Visual Results Display**: Interactive UI showing model confidence for each detected aspect

## ⚠️ Important: the trained model is not included

**`backend/pretrained_models/` is gitignored and is not present in the GitHub
repository.** There is no `.pth` checkpoint, no release artifact, and no sample
clip. The trained weights can only be produced by running the training
pipeline, which needs the licensed
[SoccerNet MVFouls dataset](https://www.soccer-net.org/data) (password by NDA).

Out of the box the API therefore starts with **randomly initialised weights**:

- the full request/response flow works and the UI renders every field,
- but **the returned decisions are meaningless**.

This state is surfaced honestly rather than hidden:

- `GET /api/health` returns `"model_trained": false`,
- every response includes `"model_trained": false`,
- the UI shows a "Demo mode: untrained model" banner.

To get real predictions, either train a checkpoint (see
[Model Training](#-model-training)) or point the app at an existing one:

```bash
set FOUL_MODEL_PATH=/path/to/foul_detection_model.pth
```

## 🏗️ Project Structure

```
backend/
  app.py                    FastAPI application and routes
  config.py                 Central, CWD-independent configuration
  inference.py              Inference pipeline (video -> features -> predictions)
  explanation.py            Referee decision text (Azure OpenAI + rule-based fallback)
  models/                   Decoder maps and action container
  utils/                    Feature extractor, preprocessor, multi-task model, HDF5 I/O
  data/loading/             SoccerNet download/explore helpers (credentialed)
  training/                 Training pipeline
  testing/                  Evaluation pipeline
  scripts/                  Development utilities
frontend/
  src/app/                  Next.js App Router entry points
  src/components/football/  Uploader, player, results, decision card
  src/lib/                  API client, Zustand store, TypeScript types
```

## 🔧 Technology Stack

**Backend**: Python, FastAPI, PyTorch, torchvision (r3d_18), OpenCV, h5py, Azure OpenAI (optional)

**Frontend**: Next.js 15, React 19, TypeScript, Zustand, Tailwind CSS v4, ShadCN UI, Axios

## 📋 Prerequisites

- **Node.js 20+** (developed against Node 24)
- **Python 3.10 – 3.13** for the pinned training stack in `requirements.txt`
  (which pins `torch==2.2.2` / `numpy==1.26.4`, and has no wheels for 3.14)
- **or Python 3.14** if you only want to *serve* the API, using
  `requirements-inference.txt` (verified working)
- A CUDA GPU is optional; the code falls back to CPU automatically
- Network access on first run to download the torchvision backbone weights
- Azure OpenAI credentials (optional — see below)
- SoccerNet credentials (only to download the dataset for training)

## 🚀 Getting Started

### Backend

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
#    Inference only (works on Python 3.14, no compiler or Java needed):
pip install -r requirements-inference.txt
#    Full training stack (Python 3.10-3.13, needs a toolchain):
pip install -r requirements.txt

# 3. Optional configuration
cp backend/.env.example backend/.env

# 4. Run it — from the repository root or from backend/, both work
python backend/app.py
```

The API is now at <http://localhost:5000> with docs at
<http://localhost:5000/docs>.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # or create it with NEXT_PUBLIC_API_URL=http://localhost:5000
npm run dev
```

Open <http://localhost:3000>.

### Optional: a clip to test with

No sample clip ships with the repository, so generate a synthetic one:

```bash
python backend/scripts/make_test_video.py
```

## ⚙️ Configuration

Every setting is an environment variable; see `backend/.env.example`. The most
useful ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FOUL_MODEL_PATH` | `backend/pretrained_models/20250129_195031/foul_detection_model.pth` | Checkpoint to load |
| `ALLOW_UNTRAINED_FALLBACK` | `true` | Serve with random weights when no checkpoint exists |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins; `*` disables CORS |
| `PORT` | `5000` | API port |
| `MAX_CONTENT_LENGTH_MB` | `100` | Upload size limit |
| `FEATURE_MODEL_TYPE` | `r3d_18` | Backbone; must match the one used in training |
| `FALLBACK_SEED` | `0` | Seed for the untrained fallback, for reproducible demos |
| `AZURE_OPENAI_*` | unset | Enable LLM-written explanations |

Azure OpenAI is entirely optional. With it unset, `explanation.py` produces a
deterministic rule-based decision from the model's severity prediction.

## 📄 API

| Endpoint | Description |
| --- | --- |
| `GET /` | Service banner and links |
| `GET /api/health` | Readiness, device, whether real weights are loaded |
| `POST /api/inference` | Analyse a clip and return a referee decision |

`POST /api/inference` takes a multipart `video` field (`mp4`, `avi`, `mov`,
`wmv`, `mkv`, `webm`; max 100 MB) and responds with:

```json
{
  "status": "success",
  "predictions": [
    {
      "category": "Severity",
      "details": [{ "prediction": "3.0 Yellow card", "probability": 0.71 }]
    }
  ],
  "ai_decision": "Yellow Card",
  "explanation": "The tackling was judged to be a foul because ...",
  "model_trained": false
}
```

## 🎯 Getting an accurate model: fine-tune the backbone

The head-only pipeline below produces a working model, but a **low-accuracy**
one. Measured on the MVFouls train split, the frozen r3d_18 features have a
between-class / within-class variance ratio of only **0.24** (action class) and
**0.32** (severity): the variation in these features is dominated by clip
appearance, not by what the foul was. Consequently every head collapses to
chance-level balanced accuracy, and the head-only model scores **21%** on the
official MVFoul metric (it answers "No card" almost always, reaching 88% plain
severity accuracy but 20% balanced accuracy).

This is structural, not a tuning problem. `r3d_18` is pretrained on
Kinetics-400, which has no foul classes, and the repo only ever trains the MLP on
top of frozen features. The reference [VARS](https://github.com/SoccerNet/sn-mvfoul)
system fine-tunes the backbone, which is where its accuracy comes from.

### On a GPU (recommended)

`notebooks/colab_finetune.ipynb` is self-contained and trains the backbone
end-to-end. In Colab: enable a **GPU runtime**, add the `MVFOULS_PASSWORD`
secret, and run all cells. It saves to Drive in a format the API reads directly.

Copy the resulting `foul_detection_model.pth` into
`backend/pretrained_models/<name>/` and the API picks it up automatically —
`inference.py` detects the fine-tuned backbone and serves with it, so predictions
use the same backbone the model was trained with.

### Locally (slow)

```bash
# Requires Python 3.10-3.13 for the pinned training stack.
pip install -r requirements.txt matplotlib
python backend/training/finetune.py --epochs 10 --batch-size 16
```

On CPU this is roughly **3 hours per epoch**, so use a GPU runtime.

### Measuring honestly

```bash
python backend/scripts/evaluate.py --split test
```

Reports per-task accuracy, the lift over each task's majority-class baseline,
per-class recall, and the two official benchmark numbers. **Judge on balanced
accuracy, not plain accuracy** — the majority-class baseline looks deceptively
strong on plain accuracy.

## 📚 Model Training

Training uses the licensed **SoccerNet-MVFouls** dataset.

### 1. Get the password

The dataset now lives on Hugging Face as
[`SoccerNet/SN-MVFouls-2024`](https://huggingface.co/datasets/SoccerNet/SN-MVFouls-2024).
The repository is public, so the **download needs no credentials**, but the
archives themselves are AES-encrypted. Request the password by signing the NDA
linked from the
[Multi-View Foul Recognition page](https://www.soccer-net.org/tasks/multi-view-foul-recognition);
they email it to you.

Set it in the environment, or use `--password`:

```bash
set MVFOULS_PASSWORD=your-password
```

### 2. Download and prepare

```bash
# valid is the smallest split (334 MB) and is a good first check.
# train is 2.3 GB low-res; train_720p is 12.7 GB and is unnecessary because
# r3d_18 resizes every clip to 224x224.
python backend/scripts/download_mvfouls.py --splits valid test train

# Normalise annotations + clip paths into data/dataset/<split>/annotations.json
python backend/scripts/prepare_dataset.py --splits valid test train --dry-run
```

`prepare_dataset.py` prints the label distribution per split, so check that
severity and action-class counts look plausible before spending compute.

### 3. Precompute features (the slow step)

```bash
python backend/scripts/precompute_features.py --splits train valid
```

One r3d_18 forward pass per clip, cached as individual `.npy` shards. **This is
resumable** — re-running skips anything already computed, so it is safe to stop
and restart. Expect roughly 1–2 hours for the full train split on CPU.

### 4. Merge and train

```bash
python backend/scripts/merge_features.py --splits train valid
python backend/training/training_pipeline.py
```

The checkpoint lands in `backend/pretrained_models/<timestamp>/`. Point the API
at it:

```bash
set FOUL_MODEL_PATH=backend/pretrained_models/<timestamp>/foul_detection_model.pth
```

### Verifying the chain without the dataset

`selftest_training.py` builds a tiny synthetic split in the exact archive layout
and runs the whole chain end to end, producing a loadable checkpoint. Useful for
confirming your setup before committing hours to feature extraction:

```bash
python backend/scripts/selftest_training.py
```

Its checkpoint is trained on synthetic clips, so it only proves the plumbing —
**its predictions are meaningless**. Delete
`backend/pretrained_models/` afterwards.

## 🧪 Verifying the installation

```bash
# Type-check and build the frontend
cd frontend
npx tsc --noEmit
npm run build

# Confirm the API is up and see whether real weights are loaded
curl http://localhost:5000/api/health

# End-to-end inference against a generated clip
python backend/scripts/make_test_video.py
curl -F "video=@backend/data/dataset/inference/testaction5_clip1.mp4" \
     http://localhost:5000/api/inference
```

## 🔍 How It Works

1. **Video upload** — the clip is streamed to a temp file with a size cap
2. **Feature extraction** — 25 frames are sampled evenly across the clip and
   passed through a Kinetics-pretrained `r3d_18` backbone
3. **Preprocessing** — pooled features are concatenated with the replay speed
   and written to HDF5, reusing the training code path
4. **Multi-task inference** — the shared trunk plus six task heads produce
   action class, body part, offence, touch ball, try to play and severity
5. **Decision generation** — the severity head maps to a card, and a
   natural-language explanation is written by Azure OpenAI or by rules
6. **Result display** — the UI renders the decision, explanation and per-task
   confidences

## 📝 Notes and known limitations

- `r3d_18` emits 512 features, so the multi-task model's `input_size` is 513
  (512 + the replay-speed scalar). Changing `FEATURE_MODEL_TYPE` without
  retraining will trip the input-size check in `inference.py`.
- `NUM_FRAMES` must stay at 25 to match the training window.
- The preprocessor keeps only the `Close-up player or field referee` camera, so
  multi-view analysis is not actually implemented despite the project name.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [SoccerNet](https://www.soccer-net.org/) for providing the MVFouls dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Next.js](https://nextjs.org/) for the frontend framework
- [ShadCN UI](https://ui.shadcn.com/) for the UI components
