# CIFAR-10 CNN Image Classifier

A CNN classifier for CIFAR-10 (10 classes: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck), written from scratch in plain PyTorch
— no pretrained weights, no transfer learning. Trained on an Apple Silicon
GPU (MPS backend).

## Architecture

VGG-style: 3 conv blocks, each with two 3x3 conv+BatchNorm+ReLU layers
followed by 2x2 max pooling, then a 2-layer classifier head with dropout.

```
input (3x32x32)
  |
  [Conv3x3(3->64)  -> BN -> ReLU]
  [Conv3x3(64->64) -> BN -> ReLU]
  MaxPool2x2                        -> 64x16x16
  |
  [Conv3x3(64->128)  -> BN -> ReLU]
  [Conv3x3(128->128) -> BN -> ReLU]
  MaxPool2x2                        -> 128x8x8
  |
  [Conv3x3(128->256) -> BN -> ReLU]
  [Conv3x3(256->256) -> BN -> ReLU]
  MaxPool2x2                        -> 256x4x4
  |
  Flatten                           -> 4096
  Dropout(0.5)
  Linear(4096 -> 512) -> ReLU
  Dropout(0.5)
  Linear(512 -> 10)
```

~3.25M parameters total.

### Design choices

- **3 conv blocks doubling channels (64/128/256)**: standard pattern for
  keeping compute roughly balanced as spatial resolution halves each block
  (32 -> 16 -> 8 -> 4). Deep enough (6 conv layers) to actually need
  BatchNorm and dropout to train well, shallow enough to stay fully
  readable and to train from scratch on a laptop in well under an hour.
  This is the intentional step up from a simpler MNIST-style CNN elsewhere
  in this portfolio — CIFAR-10's 32x32 RGB natural images with real
  intra-class variation need more capacity than MNIST digits do.
- **Two conv layers per block before pooling** (rather than one): more
  representational capacity is added at each spatial resolution before
  discarding spatial information via pooling, which matters more on small
  32x32 inputs than adding extra pooling stages would.
- **BatchNorm placed conv -> BN -> ReLU** (not conv -> ReLU -> BN):
  normalizes the pre-activation distribution so ReLU sees a consistent
  input scale throughout training. This is the original Ioffe/Szegedy
  ordering and trains faster/more stably here than the reverse.
- **Dropout only in the classifier head, not the conv blocks**: BatchNorm
  already regularizes the conv feature extractor; dropout is applied where
  the parameter count (and overfitting risk) is concentrated — the fully
  connected layers.
- **Data augmentation (random crop with 4px padding + random horizontal
  flip) on the training set only**: CIFAR-10 is small (50k training
  images at 32x32) so a plain CNN without augmentation memorizes the
  training set within a handful of epochs and its validation accuracy
  plateaus well below what augmented training reaches. Crop+flip are the
  standard, well-understood CIFAR-10 augmentations — cheap to compute and
  simulate realistic translation/mirroring variance.
- **SGD + momentum + weight decay** over Adam: the classic recipe for
  small BatchNorm CNNs on CIFAR-10, generalizes at least as well as Adam
  here and is what most reference CIFAR-10 setups use.
- **Cosine annealing LR schedule** over step decay: with a fixed, known
  epoch budget upfront, cosine annealing decays smoothly to ~0 by the
  final epoch with no milestone epochs to hand-pick, which suits a single
  short training run better than step decay's plateaus.

## Methodology

CIFAR-10 ships with 50,000 train images and 10,000 test images. A 5,000
image validation split is carved out of train (seed=42) for model
selection — the best checkpoint by validation accuracy is saved during
training — leaving the 10,000 test images completely untouched until
final evaluation. The validation split uses eval-time transforms (no
augmentation), since it exists to estimate generalization, not to be
trained on.

## Results

From the actual training run (30 epochs, batch size 128, MPS backend on an
Apple Silicon GPU):

- **Final test accuracy: 90.55%** (9,055 / 10,000)
- **Best validation accuracy: 91.22%** (epoch 30, used for checkpoint selection)
- **Final train accuracy: 94.80%** — a ~4 point train/val gap after 30
  epochs, i.e. the augmentation + dropout + BatchNorm combination is
  controlling overfitting reasonably well without eliminating it entirely
- **Training time: 52.6 minutes** for 30 epochs (~105s/epoch average;
  individual epochs ranged ~85-160s depending on system load from other
  processes running concurrently on the laptop)
- **Training curves**: `outputs/training_curves.png`
- **Confusion matrix**: `outputs/confusion_matrix.png`
- **Sample predictions**: `outputs/sample_predictions.png`

This came in noticeably above the ~70-80% range that's typical for a
briefly-trained small CNN on CIFAR-10 — the full 30-epoch cosine-annealed
schedule (rather than a shorter run) and the 6-conv-layer VGG-style depth
(rather than a shallower 2-3 layer CNN) are the main reasons why; the loss
curves below show the model was still meaningfully improving through the
low-LR tail of the schedule, not plateaued early.

The loss/accuracy curves show the classic pattern: train and val track
closely for the first ~10 epochs, then train loss pulls ahead of val loss
as the model starts fitting the training set more specifically — exactly
what the dropout and augmentation are there to slow down. Val accuracy is
noisier than train accuracy epoch-to-epoch (expected with only 5,000 val
examples vs 45,000 train), which is why checkpointing tracks the *best*
val epoch rather than just using the final one.

### Confusion matrix findings

Top confused pairs (true -> predicted, count out of 10,000 test images):

| true | predicted | count |
|---|---|---|
| cat | dog | 79 |
| dog | cat | 74 |
| airplane | ship | 33 |
| truck | automobile | 33 |
| bird | dog | 33 |
| cat | frog | 31 |
| airplane | bird | 30 |
| bird | frog | 28 |
| bird | airplane | 27 |
| bird | deer | 27 |

- **cat <-> dog is the single largest confusion by a wide margin** (79 + 74
  = 153 misclassifications between just these two classes) — this is the
  classic, expected CIFAR-10 confusion. Both are furry quadrupeds with
  similar poses, colors, and textures at 32x32 resolution, and the dataset
  itself has some genuinely ambiguous close-up shots (see the sample
  predictions grid). Cats/dogs are also the two lowest per-class recall
  classes in the matrix (cat: 798/1000, dog: 861/1000) vs. 900+ for most
  other classes.
- **truck <-> automobile (33)** and **airplane <-> ship (33)** make
  intuitive sense too: truck/car share chassis shape and road context;
  airplane/ship are both large human-made vehicles frequently photographed
  against open sky or water with a similar elongated silhouette.
  - Interestingly, automobile and ship themselves are the two easiest
    classes (967/1000 and 954/1000 recall respectively) — they get
    confused *with* their nearest visual neighbor but are otherwise very
    distinctive.
- **bird is the most scattered class**, misclassified as dog (33), frog
  (28), airplane (27), and deer (27) fairly evenly rather than concentrated
  on one confusion — birds vary enormously in pose, color, and background
  in the dataset (perched, flying, close-up, distant), so there isn't one
  dominant "bird looks like X" failure mode the way there is for cat/dog.
  Bird has the lowest per-class recall of all ten classes (858/1000).

## What this doesn't do

- No test-time augmentation (e.g. averaging predictions over flipped/
  cropped versions of each test image), which typically buys a small
  accuracy bump.
- No ensembling of multiple trained models.
- Not competitive with SOTA CIFAR-10 results (~99%+), which use much
  larger architectures (ResNets, WideResNets, etc.), heavier augmentation
  stacks (Cutout, AutoAugment, and this repo's own Mixup combined), and far longer training schedules
  (hundreds of epochs on much stronger hardware). This is a small,
  from-scratch CNN trained briefly on a laptop GPU — the goal is a
  defensible, understandable architecture and training pipeline, not a
  leaderboard result.
- No hyperparameter search — the LR, batch size, and epoch count were
  picked from standard CIFAR-10 CNN conventions and a time budget, not
  tuned via grid/random search.

## Running it

```bash
pip install -r requirements.txt
python train.py      # downloads CIFAR-10 automatically, trains, saves outputs/best_model.pth
python evaluate.py    # loads outputs/best_model.pth, evaluates on the test set
```

`outputs/best_model.pth` is gitignored (it's regenerable by rerunning
`train.py`); the plots and result JSON files in `outputs/` are committed
as evidence of the actual run.

## Checkpointing, resuming, and early stopping

```bash
python train.py --epochs 30                          # trains, writes a checkpoint every epoch
python train.py --epochs 30 --resume outputs/checkpoints/last.pth   # continue after a crash/interrupt
python train.py --epochs 30 --patience 5              # stop early if val_acc plateaus for 5 epochs
```

- `outputs/checkpoints/last.pth` is overwritten every epoch; `outputs/checkpoints/best.pth`
  only on a new best validation accuracy. Both hold model, optimizer, and
  LR-scheduler state, plus the epoch number, running history, and the
  epoch budget (`num_epochs`) that run was planned for — gitignored, since
  they're regenerable and can get large.
- **Resume is tied to the original epoch budget, not `--epochs` on the
  resume command.** The cosine LR schedule is planned for a fixed total
  epoch count set at the start of a run; if a resumed run were allowed to
  silently change that total, the restored scheduler state would desync
  from the new schedule and the LR could jump back up mid-anneal instead
  of continuing to decay. So on resume, the checkpoint's original
  `num_epochs` wins and a mismatched `--epochs` is ignored (with a printed
  note). Verified with a smoke test that interrupts a 4-epoch run after
  epoch 2, resumes it, and checks the resulting LR trajectory against an
  uninterrupted reference run — both produced exactly
  `[0.08536, 0.05, 0.01464, 0.0]`.
- **Early stopping** tracks epochs since the last validation-accuracy
  improvement and stops once that streak reaches `--patience` (default 7).
  Verified with a forced-plateau smoke test (patience=2, 6 epochs
  requested): training stopped at epoch 3, printing `early stopping:
  val_acc hasn't improved for 2 epochs (patience=2), stopping at epoch 3`.

## Mixup augmentation

```bash
python train.py --epochs 30 --mixup_alpha 0.4   # 0 or unset disables mixup (default)
```

Each training batch is mixed with a random permutation of itself —
`mixed = lam * images + (1 - lam) * images[perm]`, `lam ~ Beta(alpha, alpha)`
— and the loss is blended the same way: `lam * loss(pred, y) + (1 - lam) *
loss(pred, y[perm])`. This is deliberately **not** the common shortcut of
blending the two labels into a soft one-hot target and running plain
cross-entropy on the blend — that shortcut isn't equivalent, because
cross-entropy isn't linear in a blended one-hot target the way it is when
you blend the two per-sample losses directly (`mixup_criterion` in
`train.py`). Accuracy during a mixup epoch is measured against the
original, unpermuted label, so it's an approximation, not a strict figure,
for that epoch.

**Hardest bug: a real collapse, from a DataLoader issue, not the mixup math.**
The first full comparison run (`compare_mixup.py`, 5 epochs each,
`num_workers=4`) showed mixup completely failing to learn — loss stuck at
`ln(10) ≈ 2.303` and accuracy pinned at ~10% (random chance) for all 5
epochs, while the baseline reached 66%. That's not "mixup is a bit worse,"
that's the network stuck outputting a uniform distribution from epoch 1
onward — a strong signal something was actually broken, not just a
disappointing hyperparameter. A targeted diagnostic (same seed, same LR,
`num_workers=0` instead of 4) trained completely normally with mixup on,
which isolated the cause to the multiprocessing DataLoader workers, not
`mixup_data`/`mixup_criterion` themselves — re-running the full comparison
with `num_workers=0` gave a sane, honest result (below). The unit tests for
mixup's math (convex combination, the lambda=1 edge case, gradient flow)
had all passed the whole time, which is exactly why this was worth
tracking down rather than shrugging off as "mixup just doesn't work
here" — the math was fine; the harness feeding it data wasn't.

**Real comparative result, 5 epochs each, same seed/architecture/LR
(`compare_mixup.py`, not a full 30-epoch retrain):**

| | train_acc | val_acc |
|---|---|---|
| Baseline (no mixup) | 59.95% | 65.50% |
| Mixup (alpha=0.4) | 32.42%* | 62.36% |

*Mixup's train_acc is measured against the un-mixed label as described
above, so it understates how well the model is actually fitting the mixed
inputs — val_acc, evaluated on clean images, is the fair comparison.
Mixup trails the baseline by about 3 points here, which is the expected,
honest short-run outcome: mixup makes the training task harder from the
start (real gradient signal, blended targets, less "free" early accuracy)
in exchange for better generalization, and that trade only pays off over
longer training — 5 epochs isn't enough to see the payoff, only the cost.
This is not a "mixup helps" result; it's a "mixup works correctly and
costs what the literature says it costs at this epoch count" result.
