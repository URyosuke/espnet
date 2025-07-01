MAIN_ROOT=$PWD/../../..

# ★★★ ここが追加された最重要部分 ★★★
# PythonがESPnetのライブラリを見つけられるように、PYTHONPATHを設定
export PYTHONPATH="${MAIN_ROOT}:${PYTHONPATH:-}"
# ★★★ ここまで ★★★

# 元の行に、espnet2/bin と utils を追加
export PATH="${MAIN_ROOT}/utils:${MAIN_ROOT}/espnet2/bin:$PWD/utils/:$PATH"

export LC_ALL=C

if [ -f "${MAIN_ROOT}"/tools/activate_python.sh ]; then
    . "${MAIN_ROOT}"/tools/activate_python.sh
else
    echo "[INFO] "${MAIN_ROOT}"/tools/activate_python.sh is not present"
fi
. "${MAIN_ROOT}"/tools/extra_path.sh

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh
