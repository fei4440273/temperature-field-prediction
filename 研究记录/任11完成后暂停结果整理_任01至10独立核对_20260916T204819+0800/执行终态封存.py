#!/usr/bin/env python3
"""保存本次 CPU 只读核对的真实 stdout/stderr 与子命令退出码。"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path('/home/phl/lyf/Temperature Field Prediction')
HERE = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--批次', required=True)
    args = parser.parse_args()
    if args.批次 not in {'首次', '恢复后', '交付前', '文字修订后'}:
        raise ValueError('只允许本人固定新批次名')
    if Path.cwd() != ROOT or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU 且项目 cwd 必须显式')
    archive = HERE / (args.批次 + '_实际命令终态.json')
    if archive.exists():
        raise ValueError('不得覆盖原实际执行证据')
    script = HERE / '已有结果只读核对.py'
    data = script.read_bytes()
    command = [sys.executable, '-B', str(script), '--机器输出', str(HERE / (args.批次 + '_完整核对机器证据.json')),
               '--结果表', str(HERE / (args.批次 + '_十任务中文结果表.csv'))]
    begin = datetime.now(TZ).isoformat()
    clock = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ), text=True, capture_output=True, check=False)
    machine = {
        '执行开始_东八区': begin, '执行结束_东八区': datetime.now(TZ).isoformat(),
        '执行器PID': os.getpid(), '父PID': os.getppid(), '实际cwd': str(ROOT),
        '实际子命令argv': command, '实际子命令shell显示': shlex.join(command),
        '实际执行器argv': sys.argv, '实际子命令退出码': result.returncode,
        '完整stdout_未截断': result.stdout, '完整stderr_未截断': result.stderr,
        'CPU核对总墙钟_秒_不是训练推理成本': time.perf_counter() - clock,
        '执行时核对脚本SHA256': hashlib.sha256(data).hexdigest(),
        '执行时核对脚本全文_用于保留失败版本': data.decode('utf-8'),
        '执行时封存脚本SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        '实际环境': {key: os.environ.get(key) for key in ('PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTHONPATH', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR')},
    }
    with archive.open('x', encoding='utf-8') as stream:
        json.dump(machine, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print('真实子命令退出码=' + str(result.returncode) + '；完整终态原件=' + str(archive))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
