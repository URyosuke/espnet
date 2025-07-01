"""
D0 extractor.

Computes L2-norm of delta (first-order difference) of cepstrum features.
"""

from typing import Any, Dict, Optional, Tuple, Union

import math
import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.layers.stft import Stft
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
        ceps_order: int = 25,
        delta_K: int = 2,
        use_token_averaged_d0: bool = False,
        reduction_factor: Optional[int] = 1,
        include_power: bool = True,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.ceps_order = ceps_order
        self.delta_K = delta_K
        self.use_token_averaged_d0 = use_token_averaged_d0
        self.include_power = include_power
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

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            ceps_order=self.ceps_order,
            delta_K=self.delta_K,
            use_token_averaged_d0=self.use_token_averaged_d0,
            reduction_factor=self.reduction_factor,
            include_power=self.include_power,
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

        # 振幅スペクトル -> ケプストラム (MATLABのspec2cepsと同等の処理)
        # ceps: (B, N_frames, ceps_order)
        ceps = self._spec2ceps(input_amp, self.ceps_order)

        # ケプストラム -> Δケプストラム (MATLABのceps2dCepsと同等の処理)
        # dceps: (B, N_frames-2*K, ceps_order)
        dceps = self._ceps2dceps(ceps, self.delta_K)

        # ΔケプストラムのL2ノルム -> D0系列 (MATLABのdCeps2normと同等の処理)
        # d0: (B, N_frames-2*K)
        d0 = self._dceps2norm(dceps, self.include_power)

        # Update lengths for delta computation
        delta_stft_lengths = torch.clamp(stft_lengths - 2 * self.delta_K, min=0)

        # length adjustment
        if feats_lengths is not None:
            d0_list = [
                self._adjust_num_frames(d0_i[:l], fl)
                for d0_i, l, fl in zip(d0, delta_stft_lengths, feats_lengths)
            ]
            d0 = pad_list(d0_list, 0.0)
            d0_lengths = feats_lengths
        else:
            d0_lengths = delta_stft_lengths

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

    def _spec2ceps(self, spec: torch.Tensor, order: int) -> torch.Tensor:
        """
        スペクトル（スペクトログラム）からケプストラム（ケプストログラム）を計算
        MATLABのspec2ceps.mと同等の処理
        
        Args:
            spec: 振幅スペクトル (B, N_frames, N_freq)
            order: リフタリング次数
            
        Returns:
            ceps: ケプストラム (B, N_frames, order)
        """
        # 対数スペクトル
        log_spec = torch.log(torch.clamp(spec, min=1.0e-10))
        
        # 右半分を復元してスペクトルを左右対称に
        # log_spec: (B, N_frames, N_freq) -> (B, N_frames, 2*(N_freq-1))
        flipped_spec = torch.flip(log_spec[:, :, 1:-1], dims=[2])  # 両端を除いて反転
        symmetric_log_spec = torch.cat([log_spec, flipped_spec], dim=2)
        
        # IFFT でケプストラムを計算
        # symmetric_log_spec: (B, N_frames, 2*(N_freq-1)) -> ceps_complex: (B, N_frames, 2*(N_freq-1))
        ceps_complex = torch.fft.ifft(symmetric_log_spec, dim=2)
        ceps = ceps_complex.real
        
        # リフタリング（指定次数まで切り取り）
        ceps = ceps[:, :, :order]
        
        return ceps

    def _ceps2dceps(self, ceps: torch.Tensor, K: int) -> torch.Tensor:
        """
        ケプストラム（ケプストログラム）からΔケプストラム（の時系列）を計算
        MATLABのceps2dCeps.mと同等の処理
        
        Args:
            ceps: ケプストラム (B, N_frames, ceps_dim)
            K: 線形単回帰に用いる時間幅を決定するパラメータ
            
        Returns:
            dceps: Δケプストラム (B, N_frames-2*K, ceps_dim)
        """
        B, N_frames, ceps_dim = ceps.shape
        
        if N_frames <= 2 * K:
            # フレーム数が不足している場合はゼロを返す
            return torch.zeros(B, 0, ceps_dim, device=ceps.device, dtype=ceps.dtype)
        
        # 時間幅 k = -K:K
        k = torch.arange(-K, K+1, device=ceps.device, dtype=ceps.dtype)  # (2*K+1,)
        
        # 分母の計算: sum(k^2)
        denominator = torch.sum(k**2)  # スカラー
        
        # Δケプストラムの初期化
        dceps = torch.zeros(B, N_frames - 2*K, ceps_dim, device=ceps.device, dtype=ceps.dtype)
        
        # 各フレームについてΔケプストラムを計算
        for i, t in enumerate(range(K, N_frames - K)):  # t = K+1 to N_frames-K (0-indexed)
            # 時間窓内のケプストラム: (B, 2*K+1, ceps_dim)
            ceps_window = ceps[:, t-K:t+K+1, :]
            
            # 分子の計算: sum(k * ceps[:, t+k, :], axis=1)
            # k: (2*K+1,) -> (1, 2*K+1, 1), ceps_window: (B, 2*K+1, ceps_dim)
            k_expanded = k.unsqueeze(0).unsqueeze(2)  # (1, 2*K+1, 1)
            numerator = torch.sum(k_expanded * ceps_window, dim=1)  # (B, ceps_dim)
            
            # Δケプストラム
            dceps[:, i, :] = numerator / denominator
        
        return dceps

    def _dceps2norm(self, dceps: torch.Tensor, include_power: bool) -> torch.Tensor:
        """
        Δケプストラム（の時系列）からΔケプストラムのノルム（の時系列）を計算
        MATLABのdCeps2norm.mと同等の処理
        
        Args:
            dceps: Δケプストラム (B, N_frames, ceps_dim)
            include_power: パワー成分を含むかどうか
            
        Returns:
            norm: ノルム (B, N_frames)
        """
        # 20/log(10) でデシベル値変換の係数
        db_coeff = 20.0 / math.log(10.0)
        
        if include_power:
            # パワー成分を含む: sqrt(2*sum(dceps[2:end, :]^2) + dceps[1, :]^2)
            power_component = dceps[:, :, 0] ** 2  # (B, N_frames) - パワー成分（0次）
            other_components = torch.sum(dceps[:, :, 1:] ** 2, dim=2)  # (B, N_frames) - その他の成分
            norm_squared = power_component + 2.0 * other_components
        else:
            # パワー成分を含まない: sqrt(2*sum(dceps[2:end, :]^2))
            other_components = torch.sum(dceps[:, :, 1:] ** 2, dim=2)  # (B, N_frames)
            norm_squared = 2.0 * other_components
        
        # ノルムの計算
        norm = db_coeff * torch.sqrt(torch.clamp(norm_squared, min=1.0e-10))
        
        return norm

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
