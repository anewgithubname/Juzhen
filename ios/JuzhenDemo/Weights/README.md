Put `unet.ckpt` here: the checkpoint that `examples/demo_cnn_rectified.cu` saves
after every epoch (`RF_CHECKPOINT`, default `res/generated_cifar10/unet.ckpt`).
The app expects width 128 (`kWidth` in `CIFAR.cpp`), trained with
`RF_CH=128 RF_EPOCHS=1000` (about 7 hours on an RTX 4090, FID-10k 22).
This folder is copied into the app as is; the CIFAR-10 screen samples from it.
Checkpoints are not committed (see `ios/.gitignore`).
