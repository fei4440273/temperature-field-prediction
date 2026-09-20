"""The retained model remains the only runnable configuration."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / '研究记录/联合训练600轮_第09轮_低保真初温平滑锚定_20260920'
CONFIG = ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml'


def test_default_training_uses_identical_low_initializer_within_retained_run():
    config = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = BEST / '低保真初始化模型.pt'
    assert config['training']['epochs'] == 600
    assert config['training']['initialize_low_from'] == expected.relative_to(ROOT).as_posix()
    assert expected.is_file()
    original = json.loads((BEST / '低保真初始化记录.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(expected.read_bytes()).hexdigest() == original['检查点SHA256']
