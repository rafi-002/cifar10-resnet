# Presentation Script — CIFAR-10 ResNet

Speaker notes for the [CIFAR-10 ResNet Presentation](https://claude.ai/artifact/WrLMCWKZ9SfzuTgUGcfj6G) deck. Fill in the bracketed placeholders once training is done and you have a Kaggle score.

## 1. Cover

For this assignment, I trained a convolutional neural network from scratch on CIFAR-10 and submitted it to Kaggle's CIFAR-10 competition. The key constraint was that every weight had to be randomly initialized — no pretrained models allowed. I'll walk through the architecture, how I trained it, and the score I got.

## 2. The goal

The assignment in one line: train a CNN with random-initialized weights on the official CIFAR-10 training set, then get scored on Kaggle. Three things shaped my approach — no pretrained weights, so the model has to learn everything from 50,000 images; a score-banded rubric, where you need at least 0.88 accuracy for full marks; and it's a late submission, since the competition closed back in 2014, so my score shows up on the leaderboard as a late entry rather than a live one.

## 3. Data

I used two different data sources here. For training, I used the official CIFAR-10 dataset — 50,000 images — loaded directly through torchvision rather than Kaggle's train.7z, since the assignment explicitly allows that and it's simpler. For validation, I held out the official CIFAR-10 test set, 10,000 images, purely to estimate what my Kaggle score would be before actually submitting. The Kaggle test set itself has 300,000 images, but only 10,000 of those — the ones overlapping the original test set — actually count toward the score. The other 290,000 are distractors.

## 4. Architecture

My model is a ResNet-20, which is the CIFAR-specific version of the architecture from the original ResNet paper — not the ImageNet ResNet-18 you'd get from torchvision. It's a stem convolution followed by three stages, each with three residual blocks, going from 16 channels up to 32 then 64. In total it's about 270,000 parameters, which is small by modern standards but well suited to CIFAR's 32-by-32 images. The important part is the residual block on the right — each block does two convolutions, but also adds the original input back in before the final ReLU. That skip connection, shown in orange, is what lets you stack many layers without gradients vanishing during training — the network can always fall back to just passing the input through unchanged if that's the easiest thing to learn.

## 5. Training setup

For data augmentation I used random crop with padding and random horizontal flip — standard for CIFAR-10, and honestly the single biggest factor in pushing accuracy up. Without augmentation the model overfits and validation accuracy plateaus in the mid-80s. For the optimizer I used **[state your final choice — SGD with momentum or AdamW]** with a **[cosine annealing / OneCycle]** learning rate schedule over **[N]** epochs. Training ran on an RTX 3060 using mixed precision to fit comfortably in 6GB of VRAM. I checkpointed the model whenever validation accuracy improved, and used that held-out official test set as my estimate of the eventual Kaggle score.

## 6. Results

[Point at the curve.] Here's the training and validation accuracy over the course of training. My best training accuracy was **[X]%**, and best validation accuracy was **[Y]%** — that validation number is on the official CIFAR-10 test set, so it's a direct estimate of what I'd expect on Kaggle.

## 7. Kaggle score

After confirming the validation accuracy looked right, I generated predictions for all 300,000 Kaggle test images and submitted. Here's the leaderboard screenshot — I scored **[0.XX]**, which lines up closely with my validation estimate, as expected, since Kaggle only scores the overlapping 10,000 images.

## 8. Takeaways

A few things stood out. First, augmentation mattered more than architecture tweaks — random crop and flip closed most of the gap by themselves. Second, actually implementing the residual block by hand made the "why do skip connections help" explanation concrete instead of abstract — you can see directly how the gradient has a shortcut path back. And honestly, getting the environment set up — the right PyTorch install, the right Jupyter kernel, matching library versions — took nearly as much troubleshooting as the model code itself, which was a good reminder that the tooling is part of the job too.
