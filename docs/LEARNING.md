# Understanding CNNs & This Project — A Complete Learning Guide

---

## Part 1: What is a Neural Network?

A neural network is a function that takes numbers in and produces numbers out. It learns by looking at thousands of examples and adjusting itself to get better over time.

```
Input (pixel values of an image) → [neural network] → Output (which digit 0-9 it is)
```

### How does it learn?

1. Show it an image of "4"
2. It guesses "7"
3. You tell it "wrong, it was 4"
4. It adjusts its internal numbers (called **weights**) slightly to do better next time
5. Repeat millions of times

This adjustment process is called **backpropagation** and the algorithm that does the adjusting is called an **optimizer** (we use **Adam**).

---

## Part 2: What is a CNN?

A **Convolutional Neural Network (CNN)** is a special type of neural network designed specifically for images.

The key problem with regular neural networks on images: a 32x32 image has 3,072 numbers (32 × 32 × 3 color channels). A regular network would need billions of weights to process this. That's too slow and too easy to overfit.

CNNs solve this with **convolution**.

### What is Convolution?

A convolution is a small filter (e.g. 3x3) that slides across the entire image, doing the same math at every position.

```
Image:                Filter (3x3):         Output:
┌─────────────┐      ┌───────────┐         ┌─────────────┐
│ 0  1  0  1  │      │ 1  0  -1 │         │ edges only  │
│ 1  1  0  0  │  *   │ 1  0  -1 │    =    │ detected    │
│ 0  0  1  1  │      │ 1  0  -1 │         │ here        │
│ 1  0  0  1  │      └───────────┘         └─────────────┘
└─────────────┘
```

The filter learns to detect specific patterns:
- Early filters detect **edges and corners**
- Middle filters detect **curves and shapes**
- Deep filters detect **complex structures** like digit strokes

The key insight: the same filter is reused at every position. This means far fewer weights to learn compared to a regular network.

### What is padding=1?

When a 3x3 filter slides over an image, the edges get less coverage than the center — the output shrinks. `padding=1` adds a border of zeros around the image so the output stays the same size as the input. This is why our conv layers don't shrink the spatial dimensions — only MaxPool does.

### What is a Feature Map?

After applying a filter to an image, you get a **feature map** — a new image that highlights where that pattern was found.

```
Original image of "4"
        ↓
Vertical edge filter   → feature map showing all vertical lines
Horizontal edge filter → feature map showing all horizontal lines
Curve filter           → feature map showing all curves
```

Each conv layer produces many feature maps (one per filter). Our CustomCNN uses 32 filters in layer 1, 64 in layer 2, 128 in layer 3.

### What is Pooling?

After convolution, we apply **MaxPooling** — it shrinks the feature map by taking the maximum value in each 2x2 region.

```
Before pooling (4x4):     After MaxPool 2x2 (2x2):
┌────────────────┐        ┌────────┐
│  1   3   2   4 │        │  3   4 │
│  5   6   1   2 │   →    │  6   8 │
│  3   2   7   8 │        └────────┘
│  1   4   2   1 │
└────────────────┘
```

Why? It makes the network **location invariant** — it doesn't matter exactly where in the patch the edge was, just that it was there. It also reduces the spatial size, which reduces computation in deeper layers.

---

## Part 3: Our CustomCNN Architecture

```
Input: 32x32 RGB image (3 channels)
         │
         ▼
┌─────────────────────────────────────────┐
│  BLOCK 1                                │
│  Conv2d(3 → 32 filters, 3x3, pad=1)    │  learns 32 different edge detectors
│  BatchNorm → ReLU                       │
│  Conv2d(32 → 32 filters, 3x3, pad=1)   │  refines those edges
│  BatchNorm → ReLU                       │
│  MaxPool(2x2)  →  image is now 16x16   │
│  Dropout2d(25%)                         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  BLOCK 2                                │
│  Conv2d(32 → 64 filters, 3x3, pad=1)   │  learns 64 shape detectors
│  BatchNorm → ReLU                       │
│  Conv2d(64 → 64 filters, 3x3, pad=1)   │
│  BatchNorm → ReLU                       │
│  MaxPool(2x2)  →  image is now 8x8     │
│  Dropout2d(25%)                         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  BLOCK 3                                │
│  Conv2d(64 → 128 filters, 3x3, pad=1)  │  learns 128 complex digit-part detectors
│  BatchNorm → ReLU                       │
│  Conv2d(128 → 128 filters, 3x3, pad=1) │
│  BatchNorm → ReLU                       │
│  MaxPool(2x2)  →  image is now 4x4     │
│  Dropout2d(25%)                         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  CLASSIFIER HEAD                        │
│  Flatten  →  128 × 4 × 4 = 2048 numbers│
│  Linear(2048 → 512)                     │
│  ReLU                                   │
│  Dropout(50%)                           │
│  Linear(512 → 10)                       │  one score per digit 0-9
└─────────────────────────────────────────┘
         │
         ▼
Output: [0.02, 0.01, 0.03, 0.01, 0.87, 0.01, 0.02, 0.01, 0.01, 0.01]
         0     1     2     3     4     5     6     7     8     9
                                       ↑
                              highest score = predicted digit "4"
```

### Why these specific design choices?

**Why 3 blocks?**
Each block doubles the number of filters (32 → 64 → 128) while halving the spatial size (32 → 16 → 8 → 4). This is a standard pattern: as the image shrinks, we compensate by learning more abstract features. Fewer blocks = not enough capacity to learn digit shapes. More blocks = too slow and risks overfitting on a small dataset.

**Why two conv layers per block?**
Two consecutive conv layers without pooling in between gives the network a larger effective receptive field — it can "see" a wider area of the image before deciding what feature is there. One conv layer per block works but is less expressive.

**Why 32 → 64 → 128 filters?**
Doubling is a convention. The first layer only needs to detect simple edges (32 is enough). Deeper layers need to combine those edges into complex shapes, so they need more filters. 128 at the deepest layer is a balance between capacity and speed.

**Why Linear(2048 → 512) before the final layer?**
Going directly from 2048 to 10 is too abrupt — the network doesn't have enough capacity to combine all the spatial features into a digit decision. The 512-unit hidden layer acts as a bottleneck that learns high-level combinations.

### Key Components Explained

**BatchNorm (Batch Normalization)**
After each conv layer, the outputs can have very different scales — some neurons fire with values of 0.001, others with 100. This makes training unstable. BatchNorm rescales the outputs of each layer to have mean=0 and std=1 across the batch. This lets you use higher learning rates and makes training much more stable.

**ReLU (Rectified Linear Unit)**
The activation function: `output = max(0, input)`. It introduces non-linearity — without it, stacking layers would be pointless because the whole network would collapse to a single linear function. ReLU is preferred over older activations (sigmoid, tanh) because it doesn't suffer from the vanishing gradient problem.

**Dropout**
During training, randomly sets a percentage of neuron outputs to zero. Forces the network to not rely on any single neuron — makes it more robust. Turned off during inference (`model.eval()`). We use 25% in conv layers and 50% in the fully connected layer (higher because FC layers have more parameters and overfit more easily).

**Dropout2d vs Dropout**
`Dropout2d` zeros entire feature maps (channels) rather than individual pixels. This is better for conv layers because adjacent pixels in a feature map are highly correlated — zeroing individual pixels doesn't force the network to learn redundant representations, but zeroing whole channels does.

---

## Part 4: VGG16 — Transfer Learning

### What is VGG16?

VGG16 is a CNN architecture created by Oxford's Visual Geometry Group in 2014. It has 16 layers (13 conv + 3 FC) and was trained on **ImageNet** — 1.2 million images across 1000 categories (cats, cars, planes, etc.).

Training VGG16 from scratch took weeks on multiple GPUs. We don't need to do that.

### What is Transfer Learning?

The key insight: **the early layers of any CNN learn the same basic things** — edges, textures, shapes. These are useful for any image task, not just ImageNet.

So instead of starting from random weights, we:
1. Take VGG16's weights already trained on ImageNet
2. Keep all the conv layers (they already know how to see)
3. Replace only the final layer to output 10 classes instead of 1000
4. Fine-tune the whole network on SVHN

```
VGG16 original:                    Our VGG16:
─────────────────────              ─────────────────────
13 conv layers (ImageNet)    →     13 conv layers (kept, fine-tuned)
FC(4096) → FC(4096)          →     FC(4096) → FC(4096)
FC(4096) → FC(1000 classes)  →     FC(4096) → FC(10 classes)  ← replaced
```

### Why does this work?

A filter that detects a curved stroke in a cat's ear is the same filter useful for detecting the curve in a "6" or "9". The network already knows how to see — we just teach it what to look for.

### Why use a lower learning rate for the backbone?

In `train.py` we use two different learning rates:
```python
{"params": model.features.parameters(),   "lr": args.lr * 0.1},  # 1e-5
{"params": model.classifier.parameters(), "lr": args.lr},         # 1e-4
```

The pretrained conv layers already have good weights — we don't want to destroy them with large updates. The new classifier head starts from random weights and needs larger updates to learn quickly. This technique is called **differential learning rates**.

### Why not freeze the backbone entirely?

We could freeze the conv layers and only train the classifier head. This is faster but less accurate — the ImageNet features aren't perfectly suited for SVHN digits. Allowing the backbone to fine-tune (with a small lr) lets it adapt its features to digit-specific patterns while preserving the general visual knowledge.

---

## Part 5: The Dataset

### SVHN Format 1

The Street View House Numbers dataset contains real photos of house numbers taken from Google Street View.

```
data/train/
├── 1.png          ← full street photo (e.g. house with "42" on it)
├── 2.png
├── ...
└── digitStruct.mat   ← ground truth: for every image, what digits are where
```

`digitStruct.mat` is a MATLAB file that for each image stores:
- `labels` — what digit each bounding box contains (1-9, and 10 means '0')
- `top`, `left`, `height`, `width` — the bounding box coordinates

### Why label 10 = digit '0'?

MATLAB arrays are 1-indexed. SVHN uses labels 1-10 where 1-9 are digits 1-9 and 10 represents digit 0. We remap `10 → 0` in `dataset.py` so our model uses labels 0-9.

### What dataset.py does

```
digitStruct.mat
    ↓ parse_digit_struct()
list of {filename, labels, tops, lefts, heights, widths}
    ↓ SVHNFormat1Dataset.__getitem__()
open image → crop bounding box → clamp to image bounds → resize to 32x32
    ↓ transforms
normalize pixel values → tensor
    ↓
(32x32 tensor, label) — one training sample
```

### Why clamp bounding boxes?

Some bounding boxes in digitStruct.mat extend slightly outside the image boundaries (annotation errors). Without clamping, `img.crop()` would either crash or return an empty crop. We clamp `x1, y1` to 0 and `x2, y2` to the image width/height.

### Train / Val / Test split

- **Train**: 73,257 digit crops — used to update weights
- **Val**: 10% of train (randomly split) — used to monitor overfitting, never used to update weights
- **Test**: 26,032 digit crops — used only once at the end to report final accuracy

The val set uses a separate dataset instance with no augmentation. This is important — if val used augmentation, the val loss would be artificially higher (augmented images are harder) and wouldn't accurately reflect how well the model generalises.

### Data Augmentation

Applied only to the training set:
```python
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)  # simulate lighting variation
RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))  # simulate pose variation
```

Why? The model will see real-world images with varying lighting and slight rotations. Augmentation artificially creates these variations during training so the model learns to handle them. Without augmentation, the model would overfit to the clean SVHN crops and fail on messier real-world images.

### Normalisation

```python
transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
```

These are the per-channel mean and standard deviation of the SVHN dataset. Normalising subtracts the mean and divides by std so pixel values are centred around 0. This makes gradient descent more stable — without it, the loss landscape is skewed and training is slower.

---

## Part 6: Training

### Loss Function — CrossEntropyLoss

After the model outputs 10 scores, we need to measure how wrong it was. CrossEntropyLoss does this:

```
True label: 4
Model output: [0.02, 0.01, 0.03, 0.01, 0.87, 0.01, 0.02, 0.01, 0.01, 0.01]

Loss = -log(0.87) = 0.14   ← small loss, model was mostly right

If model output: [0.02, 0.01, 0.03, 0.01, 0.10, 0.01, 0.02, 0.70, 0.01, 0.01]
Loss = -log(0.10) = 2.30   ← large loss, model was very wrong
```

CrossEntropyLoss internally applies **Softmax** first (converts raw scores to probabilities that sum to 1), then computes the negative log of the probability assigned to the correct class. The optimizer then uses this loss to adjust weights — bigger loss = bigger adjustment.

### Optimizer — Adam

Adam is an adaptive learning rate optimizer. It keeps a running estimate of both the gradient and the squared gradient for each weight, and uses these to scale the update for each weight individually. Weights that have been updated a lot get smaller steps; weights that haven't been updated much get larger steps.

**Learning rate** controls the overall step size:
- Too high → overshoots the minimum, training is unstable or diverges
- Too low → takes forever to converge
- We use `1e-3` for CustomCNN, `1e-4` for VGG16 (smaller because pretrained weights are already close to a good solution)

**Weight decay (1e-4)** adds a small penalty for large weights. This is L2 regularisation — it prevents any single weight from becoming too large, which helps generalisation.

### Learning Rate Scheduler

```python
ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
```

If the validation loss doesn't improve for 3 epochs, the learning rate is halved. This lets training start with larger steps (fast progress) and automatically slow down when it gets close to a good solution (fine-grained refinement).

### Batch Size

Instead of updating weights after every single image, we process a **batch** of images at once and average the loss.
- Larger batch (128 for CustomCNN) → more stable gradient estimate, faster on GPU
- Smaller batch (64 for VGG16) → VGG16 is much larger, 128 would run out of memory

### Early Stopping

We monitor validation loss after every epoch. If it hasn't improved in 7 epochs, we stop training. This prevents **overfitting**.

```
Epoch 1:  val_loss=1.20  ← save checkpoint
Epoch 2:  val_loss=0.95  ← save checkpoint
Epoch 3:  val_loss=0.80  ← save checkpoint
Epoch 4:  val_loss=0.82  ← no improvement (counter: 1)
Epoch 5:  val_loss=0.85  ← no improvement (counter: 2)
...
Epoch 10: val_loss=0.91  ← no improvement (counter: 7) → STOP
```

Best checkpoint (epoch 3) is what gets saved and used for inference.

### What gets saved

- `checkpoints/custom_best.pth` / `checkpoints/vgg16_best.pth` — model weights at best val loss
- `results/custom_curves.png` — loss and accuracy plots
- `results/custom_history.json` — all epoch metrics + final test loss/acc

### model.train() vs model.eval()

PyTorch models have two modes:
- `model.train()` — Dropout is active (randomly zeros neurons), BatchNorm uses batch statistics
- `model.eval()` — Dropout is disabled (all neurons active), BatchNorm uses running statistics

We switch to `model.eval()` during validation and inference. Forgetting this is a common bug — the model would give different results every time you run it due to random dropout.

### @torch.no_grad()

During evaluation we don't need to compute gradients (we're not updating weights). `@torch.no_grad()` disables gradient tracking, which saves memory and speeds up inference by ~2x.

---

## Part 7: The Detection Pipeline

Training gives us a model that classifies a single 32x32 crop. But at test time we have a full photo — we don't know where the digits are. This is the **detection** problem.

### Step 1 — Preprocessing

**Gaussian Blur**
```python
cv2.GaussianBlur(img, (3, 3), 0)
```
A 3x3 Gaussian kernel averages each pixel with its neighbours, weighted by a Gaussian distribution. This smooths out pixel-level noise (camera grain, JPEG compression artifacts) without blurring large structures like digit edges. We use a small kernel (3x3) to preserve detail.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
```python
cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```
Regular histogram equalisation stretches the contrast of the whole image globally. CLAHE divides the image into 8x8 tiles and equalises each tile independently. This means a dark corner gets brightened without overexposing an already bright area. Applied only to the L (lightness) channel in LAB colour space so colours aren't distorted. `clipLimit=2.0` prevents over-amplifying noise in uniform regions.

### Step 2 — Region Proposals

We don't know where digits are, so we generate thousands of candidate boxes.

**MSER (Maximally Stable Extremal Regions)**
Imagine thresholding the image at every possible brightness level (0, 1, 2, ... 255). At each threshold, connected regions of pixels appear and grow. MSER finds regions that stay stable (don't change much) across a range of thresholds. Text and digits are naturally stable because they have consistent contrast against their background.

```python
cv2.MSER_create(min_area=100, max_area=5000)
```
- `min_area=100` — ignore tiny noise blobs
- `max_area=5000` — ignore large background regions
- Aspect ratio filter (0.2 < w/h < 5.0) — digits are roughly square, not extremely wide or tall

**Sliding Window**
A 32x32 window slides across the image in steps of 16 pixels (50% overlap), proposing every position as a candidate. The 50% overlap ensures no digit is missed between steps.

**Image Pyramid**
The digit could be any size. We shrink the image by 75% repeatedly and run the sliding window at each scale. A small digit in the original becomes a normal-sized digit in the shrunken version. Proposals are mapped back to original coordinates by dividing by the scale factor.

```
Original (512x256) → proposals at scale 1.0
Shrink to 75%      → proposals mapped back: coords / 0.75
Shrink to 56%      → proposals mapped back: coords / 0.56
...
```

### Step 3 — Classify Each Box

```python
for each candidate box:
    crop it from the preprocessed image
    resize to 32x32
    convert BGR → RGB (OpenCV uses BGR, PyTorch expects RGB)
    normalise pixel values (same normalisation as training)
    run through trained CNN in batches of 64
    apply softmax → probabilities
    if max probability > 0.85 → keep this detection
```

**Why 0.85 confidence threshold?**
The model has no explicit "background" class — it always outputs a digit. The confidence threshold is how we distinguish real digits from background patches that happen to look vaguely like a digit. 0.85 means the model must be very sure. Lower = more false positives. Higher = more missed detections.

**Why batch size 64 for inference?**
Processing all proposals one at a time would be very slow. Batching 64 at a time lets the GPU process them in parallel. 64 is a balance between speed and memory.

### Step 4 — Non-Maximum Suppression (NMS)

After classification we have hundreds of overlapping boxes all claiming to be the same digit. NMS cleans this up.

**Why per-class NMS?**
We run NMS separately for each digit class (0-9). This prevents a "4" box from suppressing a nearby "2" box just because they overlap. Two different digits can legitimately be close together.

```
1. Sort all boxes for class X by confidence (highest first)
2. Take the highest confidence box → keep it
3. Remove all other boxes that overlap it by more than 30% (IoU > 0.3)
4. Repeat with the next highest remaining box
5. Until no boxes left
```

**IoU (Intersection over Union)**
```
IoU = area of overlap / area of union

IoU = 0.0  → boxes don't overlap at all
IoU = 1.0  → boxes are identical
IoU = 0.3  → our threshold
```

Why 0.3? Digits are small and close together. A stricter threshold (e.g. 0.5) would keep too many duplicates. A looser threshold (e.g. 0.1) would suppress legitimate nearby detections.

### Step 5 — Assemble the Digit Sequence

```python
detections sorted by x coordinate (left → right)
→ read off labels → "4", "2" → "42"
```

This assumes digits are written left to right, which is true for Western numerals. For other writing systems this would need to change.

---

## Part 8: The Full Picture

```
┌─────────────────────────────────────────────────────────────┐
│                        TRAINING                             │
│                                                             │
│  train.tar.gz                                               │
│      ↓ extract                                              │
│  33,402 full street photos + digitStruct.mat                │
│      ↓ dataset.py: parse bboxes, crop digits, resize 32x32 │
│  73,257 labelled digit crops                                │
│      ↓ train.py: batches → model → loss → backprop → Adam  │
│        repeat for up to 30 epochs, early stop at patience=7 │
│  checkpoints/vgg16_best.pth                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        INFERENCE (run.py)                   │
│                                                             │
│  new photo (e.g. house with "42" on it)                     │
│      ↓ Gaussian blur + CLAHE                                │
│      ↓ MSER + sliding window + image pyramid                │
│  ~1000 candidate boxes                                      │
│      ↓ crop each → resize 32x32 → CNN → softmax            │
│  filter: confidence > 0.85                                  │
│      ↓ per-class NMS (IoU threshold 0.3)                    │
│  clean detections                                           │
│      ↓ sort left → right by x                              │
│  "42"  +  annotated image → graded_images/1.png            │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 9: The Two Models Compared

| | CustomCNN | VGG16 |
|---|---|---|
| Starting weights | Random | ImageNet pretrained |
| Parameters | ~2M | ~138M |
| Training epochs | ~30 | ~20 |
| Learning rate | 1e-3 | 1e-4 (backbone), 1e-3 (head) |
| Batch size | 128 | 64 |
| Expected accuracy | ~90-92% | ~95-97% |
| Training time (CPU) | ~2-3 hrs | ~4-6 hrs |
| Use case | Lightweight | High accuracy |

VGG16 almost always wins on accuracy because it starts from a much better initialisation. CustomCNN is useful to understand what you can achieve training from scratch.

---

## Part 10: Why Things Can Go Wrong

| Problem | Cause | Effect |
|---|---|---|
| Digit too small | Pyramid didn't go small enough | Missed detection |
| Digit at steep angle | RandomAffine only goes to ±15° | Wrong classification |
| Poor lighting | CLAHE didn't fully correct | Low confidence, missed |
| Digits touching | NMS merges them | Two digits read as one |
| Background texture | Looks like a digit to CNN | False positive |
| Confidence threshold too high | 0.85 filters real digits | Missed detection |
| Confidence threshold too low | Keeps background patches | False positives |
| Train/test domain gap | SVHN crops vs real photos | Lower real-world accuracy |
| Overfitting | Model memorised training data | High train acc, low test acc |

These failure cases are exactly what your report needs to analyse — show images where it fails and explain why.

---

## Part 11: Key Terms Glossary

| Term | Meaning |
|---|---|
| **Weight** | A number inside the network that gets adjusted during training |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | A small group of images processed together before updating weights |
| **Loss** | A number measuring how wrong the model's predictions are |
| **Gradient** | The direction and magnitude to adjust each weight to reduce loss |
| **Backpropagation** | Algorithm that computes gradients through every layer of the network |
| **Overfitting** | Model memorises training data, fails on new unseen data |
| **Underfitting** | Model is too simple to learn the patterns in the data |
| **Dropout** | Randomly disabling neurons during training to prevent overfitting |
| **Dropout2d** | Randomly disabling entire feature map channels (better for conv layers) |
| **BatchNorm** | Normalises layer outputs to have mean=0, std=1 — stabilises training |
| **ReLU** | Activation function: max(0, x) — introduces non-linearity |
| **Softmax** | Converts raw scores to probabilities that sum to 1 |
| **CrossEntropyLoss** | Loss function for classification: -log(probability of correct class) |
| **Adam** | Adaptive optimizer that adjusts learning rate per weight |
| **Weight decay** | L2 regularisation penalty on large weights — prevents overfitting |
| **Learning rate** | How big each weight update step is |
| **Scheduler** | Automatically reduces learning rate when progress stalls |
| **Early stopping** | Stop training when val loss stops improving |
| **Transfer learning** | Reusing weights trained on one task for a different task |
| **Fine-tuning** | Continuing to train pretrained weights on a new dataset |
| **Feature map** | Output of a conv layer — highlights where a pattern was found |
| **Receptive field** | The area of the original image a neuron can "see" |
| **Padding** | Zeros added around image borders so conv output stays same size |
| **IoU** | Intersection over Union — measures how much two boxes overlap |
| **NMS** | Non-Maximum Suppression — removes duplicate detections |
| **MSER** | Maximally Stable Extremal Regions — finds text-like blobs in images |
| **CLAHE** | Adaptive histogram equalisation for lighting correction |
| **Image pyramid** | Repeatedly shrinking an image to detect objects at multiple scales |
| **Sliding window** | Scanning every position of an image with a fixed-size crop |
| **Confidence threshold** | Minimum probability required to accept a detection |
| **Differential lr** | Using different learning rates for different parts of the network |
| **Augmentation** | Artificially creating training variation (rotation, brightness, etc.) |
| **Normalisation** | Scaling pixel values to have mean=0, std=1 for stable training |
