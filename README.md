# pl_epoch_invariance_in_cv
`pytorch-lightning` を用いたときに，エポック数を変化させても学習結果が重なるようにする方法

# `check_val_every_n_epoch` を変えると `dataloader` の出力が変わる理由

## 前提
乱数を固定しても，サイコロを2回ふると異なる結果が得られる．
```Python:fix_seed_but_different_results.py
pl.seed_everything(42)
print(torch.rand(1)) # tensor([0.8823])
print(torch.rand(1)) # tensor([0.9150])
```
これは `torch.rand(1)` によって乱数が更新されるためである．

## 実際の流れを追う
`validation_step()` で乱数が更新されるため，2回目の`training_step()`から結果が変わってしまう．

### `epoch_val_every_n_epoch=1` の場合
```
training_step() x iterations # 乱数更新
↓
validation_step() # 乱数更新
↓
training_step() x iterations # 乱数更新
↓
validation_step() # 乱数更新
↓
training_step() x iterations # 乱数更新
↓
validation_step() # 乱数更新
↓
training_step() x iterations # 乱数更新
↓
validation_step() # 乱数更新
```

### `epoch_val_every_n_epoch=2` の場合
```
training_step() x iterations # 乱数更新
↓
training_step() x iterations # 乱数更新
↓
validation_step() # 乱数更新
↓
training_step() x iterations # 乱数更新
↓
training_step() x iterations # 乱数更新
↓
validation_step() # 乱数更新
```

# 対策
`dataloader` の `generator` を固定することで `dataloader` 専用の乱数の生成器を設定できる．`validation_step()` による乱数の更新に依存せず，`dataloader` のみの乱数を使える．
