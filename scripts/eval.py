import numpy as np
import pandas as pd
import soundfile as sf
import os

import torch

from torchmetrics.functional import (
    permutation_invariant_training as PIT,  # noqa
    scale_invariant_signal_noise_ratio as SISNR,  # noqa
    signal_distortion_ratio as SDR  # noqa
)

from speechbrain.pretrained import SepformerSeparation

from tqdm.auto import tqdm


def to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def build_model() -> torch.nn.Module:
    source = 'speechbrain/sepformer-libri2mix'
    savedir = 'pretrained_models/sepformer-libri2mix'

    model = SepformerSeparation.from_hparams(source=source, savedir=savedir)
    model.eval()
    return model


def main():
    savepath = '/home/alexnikko/prog/icassp/eval_results/wav8k_min_test_mix_clean_results.csv'
    path_to_test_meta = '/home/alexnikko/data/Libri2Mix/wav8k/min/metadata/mixture_test_mix_clean.csv'
    meta = pd.read_csv(path_to_test_meta)

    device = 'cuda:0'

    model = build_model()
    model.to(device)
    model.device = device  # SepformerSeparation ugly property =/

    metrics = []
    for row in tqdm(meta.itertuples(), total=len(meta)):
        s1, _ = sf.read(row.source_1_path, dtype='float32')
        s2, _ = sf.read(row.source_2_path, dtype='float32')
        mixture, _ = sf.read(row.mixture_path, dtype='float32')

        s1, s2, mixture = [to_torch(x).float() for x in [s1, s2, mixture]]
        target = torch.stack((s1, s2)).unsqueeze(0)  # batch, num_spk, T

        with torch.inference_mode():
            preds = model(mixture.unsqueeze(0).to(device))  # batch, T, num_spk
        preds = preds.cpu().permute(0, 2, 1)

        sisnr, perm = PIT(preds, target, metric_func=SISNR, eval_func='max')
        sisnr_inp = SISNR(mixture[None, None].expand_as(target), target).mean()

        sdr, perm = PIT(preds, target, metric_func=SDR, eval_func='max')
        sdr_inp = SDR(mixture[None, None].expand_as(target), target).mean()

        metrics.append({
            'SI-SNR_inp': sisnr_inp.item(),
            'SI-SNR': sisnr.item(),
            'SI-SNRi': sisnr.item() - sisnr_inp.item(),
            'SDR_inp': sdr_inp.item(),
            'SDR': sdr.item(),
            'SDRi': sdr.item() - sdr_inp.item(),
        })
        break

    metrics_df = pd.DataFrame(metrics)

    # print mean values
    for key, value in metrics_df.mean().to_dict().items():
        print(f'{key} = {value}')

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    metrics_df.to_csv(savepath, index=False)


if __name__ == '__main__':
    main()
