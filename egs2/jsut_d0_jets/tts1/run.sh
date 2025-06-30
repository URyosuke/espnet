#!/usr/bin/env bash
set -e
set -u
set -o pipefail

sampling_rate=22.05k

# サンプリング周波数・FFTポイント・シフト長
if [ "${sampling_rate}" = 48k ]; then
    fs=48000
    n_fft=2048
    n_shift=512
elif [ "${sampling_rate}" = 44.1k ]; then
    fs=44100
    n_fft=2048
    n_shift=512
elif [ "${sampling_rate}" = 24k ]; then
    fs=24000
    n_fft=1024
    n_shift=256
elif [ "${sampling_rate}" = 22.05k ]; then
    fs=22050
    n_fft=1024
    n_shift=256
fi

# オーディオフォーマット
if [ "${fs}" -eq 48000 ]; then
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

# 訓練・検証・テストセットのディレクトリ名
train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

# dump・expディレクトリ名(dumpディレクトリは上の検証データ等を格納するディレクトリ)
dumpdir="dump/${sampling_rate}"
expdir="exp/${sampling_rate}"

# 設定ファイル
train_config=conf/tuning/train_jets_${sampling_rate}.yaml
inference_config=

# 書記素から音素への変換方法
g2p=pyopenjtalk_prosody

# オプション
opts+="--min_wav_duration 0.8 "       # 秒単位の最小持続時間
opts+="--tts_task gan_tts "           # GANベースのTTSモデルを使用
opts+="--write_collected_feats true " # 統計収集内の特徴量をdump
opts+="--dumpdir "${dumpdir}" "       # dumpディレクトリ名
opts+="--expdir "${expdir}" "         # expディレクトリ名
opts+="--inference_model latest.pth " # 推論時に使用するモデルのパス

# 実行
./tts.sh \
    --lang jp \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type phn \
    --cleaner jaconv \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
