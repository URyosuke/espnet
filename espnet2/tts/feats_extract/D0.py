"""
D0 extractor.

Computes L2-norm of delta (first-order difference) of mel-cepstrum features.
"""

from typing import Any, Dict, Optional, Tuple, Union

import math
import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.layers.stft import Stft
from espnet2.layers.log_mel import LogMel
from espnet.nets.pytorch_backend.nets_utils import pad_list

class D0(AbsFeatsExtract):
    """D0 extractor."""
    
    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: Optional[float] = None,
        use_token_averaged_d0: bool = False,
        reduction_factor: Optional[int] = 1,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.use_token_averaged_d0 = use_token_averaged_d0
        if use_token_averaged_d0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        
        # 対数メルスペクトログラムの計算
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            log_base=log_base,
        )
        # create DCT-II matrix for mel-cepstrum
        n = torch.arange(n_mels).unsqueeze(1).float()
        k = torch.arange(n_mels).unsqueeze(0).float()
        dct_mat = torch.cos(math.pi * k * (2 * n + 1) / (2 * n_mels))
        self.register_buffer("dct_mat", dct_mat)

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            win_length=self.win_length,
            fmin=self.logmel.mel_options["fmin"],
            fmax=self.logmel.mel_options["fmax"],
            use_token_averaged_d0=self.use_token_averaged_d0,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        feats_lengths: torch.Tensor = None,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If not provided, assume same length
        if input_lengths is None:
            input_lengths = input.new_ones(input.size(0), dtype=torch.long) * input.size(1)

        # STFT: 音声波形 -> 複素スペクトログラム
        # input: (B, T_audio) -> input_stft: (B, N_frames, N_freq, 2), stft_lengths: (B,)
        input_stft, stft_lengths = self.stft(input, input_lengths)
        assert input_stft.dim() >= 4, input_stft.shape  # 4次元以上であることを確認
        assert input_stft.shape[-1] == 2, input_stft.shape  # 複素数(実部・虚部)であることを確認

        # 複素スペクトログラム -> パワースペクトル -> 振幅スペクトル
        # input_power: (B, N_frames, N_freq), input_amp: (B, N_frames, N_freq)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))

        # 振幅スペクトル -> 対数メルスペクトログラム
        # logmel_feats: (B, N_frames, n_mels), mel_lengths: (B,)
        logmel_feats, mel_lengths = self.logmel(input_amp, stft_lengths)

        # 対数メルスペクトログラム -> メルケプストラム (DCT変換)
        # mcep: (B, N_frames, n_mels)
        mcep = torch.matmul(logmel_feats, self.dct_mat)

        # メルケプストラム -> 一次差分 (Δ特徴量)
        # diff: (B, N_frames-1, n_mels), delta: (B, N_frames, n_mels)
        diff = mcep[:, 1:, :] - mcep[:, :-1, :]  # 隣接フレーム間の差分
        zero = mcep.new_zeros(mcep.size(0), 1, mcep.size(2))  # 先頭フレーム用のゼロ
        delta = torch.cat([zero, diff], dim=1)  # 時間軸で結合

        # Δ特徴量のL2ノルム -> D0系列
        # d0: (B, N_frames)
        d0 = torch.norm(delta, dim=2)  # 各フレームの特徴量次元でL2ノルム

        # length adjustment
        if feats_lengths is not None:
            d0_list = [
                self._adjust_num_frames(d0_i[:l], fl)
                for d0_i, l, fl in zip(d0, mel_lengths, feats_lengths)
            ]
            d0 = pad_list(d0_list, 0.0)
            d0_lengths = feats_lengths
        else:
            d0_lengths = mel_lengths

        # token-level average
        if self.use_token_averaged_d0:
            durations = durations * self.reduction_factor
            d0_list = [
                self._average_by_duration(d0_i[:l], d)
                for d0_i, l, d in zip(d0, d0_lengths, durations)
            ]
            d0 = pad_list(d0_list, 0.0)
            d0_lengths = durations_lengths

        # 最終出力: (B, T, 1) 形状に整形
        # d0: (B, T, 1), d0_lengths: (B,)
        return d0.unsqueeze(-1), d0_lengths

    def _average_by_duration(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert 0 <= len(x) - d.sum() < self.reduction_factor
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x
