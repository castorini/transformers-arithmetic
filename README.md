# transformers-arithmetic

This repository contains the code to reproduce the experiments from the paper ["Investigating the Limitations of Transformers with Simple Arithmetic Tasks"](https://arxiv.org/abs/2102.13019)

First, install the required packages:
```
pip install -r requirements.txt
```

The command below trains and evaluates a T5-base model on the task of adding up to 15-digits:

```
python main.py \
    --output_dir=. \
    --model_name_or_path=t5-base \
    --operation=addition \
    --orthography=10ebased \
    --balance_train \
    --balance_val \
    --train_size=100000 \
    --val_size=10000 \
    --test_size=10000 \
    --min_digits_train=2 \
    --max_digits_train=15 \
    --min_digits_test=2 \
    --max_digits_test=15 \
    --base_number=10 \
    --seed=1 \
    --train_batch_size=4 \
    --accumulate_grad_batches=32 \
    --val_batch_size=32 \
    --max_seq_length=512 \
    --num_workers=4 \
    --gpus=1 \
    --optimizer=AdamW \
    --lr=3e-4 \
    --weight_decay=5e-5 \
    --scheduler=StepLR \
    --t_0=2 \
    --t_mult=2 \
    --gamma=1.0 \
    --step_size=1000 \
    --max_epochs=20 \
    --check_val_every_n_epoch=2 \
    --amp_level=O0 \
    --precision=32 \
    --gradient_clip_val=1.0
```

This training should take approximately 10 hours on a V100 GPU.
The exact match on the test set should be 1.