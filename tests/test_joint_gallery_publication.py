"""固定图册发布检查；合成指标和图像只用于程序测试。"""
import base64
import hashlib
import io
import json
import shutil
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import joint_temperature_figures as gallery
from joint_temperature_core import Geometry, JointDeepONet, PhysicalSettings, fixed_splits

FIXED = Path('研究记录/联合训练8000轮_20260917_114704/结果总览.html')


@pytest.fixture
def project():
    directory = Path(tempfile.mkdtemp(prefix='临时图册测试_', dir=ROOT / '研究记录'))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def file_hashes(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def metrics(error):
    return {name: {'逐功率': [{'功率_W': 169., '均方根误差_℃': error, '平均绝对误差_℃': error / 2}]}
            for name in ('顶部', '热端', '冷端')}


def complete_run(project, name='联合训练8000轮_测试 空格', error=3.):
    out = project / '研究记录' / name
    (out / '训练曲线').mkdir(parents=True, exist_ok=True)
    (out / '169.000W/表面逐时刻').mkdir(parents=True, exist_ok=True)
    for name in ('训练曲线/损失函数变化.png', '169.000W/热端温度对比.png',
                 '169.000W/表面逐时刻/0005.000秒_并排对比.png'):
        Image.new('RGB', (10, 6), (20, 100, 180)).save(out / name)
    Image.new('RGB', (10, 6), (180, 100, 20)).save(out / '169.000W/整体三维温度_实验校正预测.gif')
    checkpoint = out / '验证最佳模型.pt'
    checkpoint.write_bytes(b'unit test checkpoint, not a trained model')
    lock = {'完成轮数': 8000, '出图检查点': checkpoint.name,
            '检查点SHA256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            '时间': '2026-09-18T00:50:31+08:00'}
    (out / '测试出图锁定记录.json').write_text(json.dumps(lock, ensure_ascii=False), encoding='utf-8')
    values = metrics(error)
    (out / '测试指标_原始精度.json').write_text(json.dumps(values, ensure_ascii=False), encoding='utf-8')
    gallery.write_gallery(out, values)
    return out


class Images(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.sources = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        if tag == 'img':
            self.sources.append(dict(attributes)['src'])


def assert_assets_from(destination, out, *, embedded=False, expected_count=4):
    sources = Images(destination.read_text(encoding='utf-8')).sources
    if expected_count is not None:
        assert len(sources) == expected_count
    if embedded:
        original = {path.read_bytes() for path in out.rglob('*') if path.suffix in ('.png', '.gif')}
        for source in sources:
            assert source.startswith(('data:image/png;base64,', 'data:image/gif;base64,'))
            header, encoded = source.split(',', 1)
            data = base64.b64decode(encoded, validate=True)
            assert data in original
            with Image.open(io.BytesIO(data)) as image:
                assert header == f'data:{Image.MIME[image.format]};base64'
        return
    for source in sources:
        parsed = urlsplit(source)
        assert not parsed.scheme and not parsed.netloc
        asset = (destination.parent / unquote(parsed.path)).resolve()
        assert asset.is_file() and out.resolve() in asset.parents


def test_existing_write_gallery_two_argument_api_is_unchanged(project):
    out = complete_run(project)
    assert_assets_from(out / '结果总览.html', out)


def test_fixed_gallery_embeds_original_png_and_animated_gif_bytes(project):
    out = complete_run(project)
    gif = out / '169.000W/整体三维温度_实验校正预测.gif'
    frames = [Image.new('RGB', (10, 6), color) for color in ((180, 100, 20), (20, 100, 180))]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=200, loop=0)
    for frame in frames:
        frame.close()
    expected = sorted(path.read_bytes() for path in out.rglob('*') if path.suffix in ('.png', '.gif'))
    fixed = gallery.publish_latest_gallery(project, out)
    sources = Images(fixed.read_text(encoding='utf-8')).sources
    assert len(sources) == 4
    decoded = []
    for source in sources:
        assert source.startswith(('data:image/png;base64,', 'data:image/gif;base64,'))
        header, encoded = source.split(',', 1)
        data = base64.b64decode(encoded, validate=True)
        decoded.append(data)
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (10, 6)
            assert header == f'data:{Image.MIME[image.format]};base64'
            if image.format == 'GIF':
                assert image.n_frames == 2
                image.seek(1)
                image.load()
    assert sorted(decoded) == expected


def test_fixed_gallery_images_remain_readable_without_the_source_directory(project):
    out = complete_run(project)
    fixed = gallery.publish_latest_gallery(project, out)
    out.rename(out.with_name('单元测试已移动来源目录'))
    sources = Images(fixed.read_text(encoding='utf-8')).sources
    assert len(sources) == 4
    for source in sources:
        assert source.startswith('data:image/')
        with Image.open(io.BytesIO(base64.b64decode(source.split(',', 1)[1], validate=True))) as image:
            image.load()
            assert image.size == (10, 6)


@pytest.fixture
def surface_frames():
    return [(10., np.array([[300., 200.], [np.nan, 50.]]),
             np.array([[303., 196.], [np.inf, np.nan]]), [0., 1., 0., 1.], '实验温度'),
            (5., np.array([[20., 20.], [20., np.nan]]),
             np.array([[19., 23.], [20., np.nan]]), [0., 1., 0., 1.], '实验温度')]


def test_surface_error_rows_use_matching_finite_cloud_pixels_and_sorted_times(surface_frames):
    assert hasattr(gallery, 'surface_error_rows'), '顶面云图缺少逐时刻误差计算'
    rows = gallery.surface_error_rows(surface_frames)
    assert [row['时间_s'] for row in rows] == [5., 10.]
    assert [row['有效点数'] for row in rows] == [3, 2]
    assert rows[0]['均方根误差_℃'] == pytest.approx(np.sqrt(10. / 3.))
    assert rows[0]['平均绝对误差_℃'] == pytest.approx(4. / 3.)
    assert rows[0]['平均偏差_℃'] == pytest.approx(2. / 3.)
    assert rows[1]['均方根误差_℃'] == pytest.approx(np.sqrt(12.5))
    assert rows[1]['平均绝对误差_℃'] == pytest.approx(3.5)
    assert rows[1]['平均偏差_℃'] == pytest.approx(-.5)
    assert rows[1]['最大绝对误差_℃'] == 4.
    assert rows[1]['数据来源'] == '实验温度'


def test_surface_bias_retains_its_sign_without_cancelling_absolute_errors():
    assert hasattr(gallery, 'surface_error_rows'), '顶面云图缺少逐时刻误差计算'
    frames = [(5., np.array([20., 20.]), np.array([16., 24.]), [], '实验环平均数据还原')]
    row = gallery.surface_error_rows(frames)[0]
    assert row['平均偏差_℃'] == 0.
    assert row['平均绝对误差_℃'] == row['均方根误差_℃'] == 4.


@pytest.mark.parametrize('invalid', ['empty', 'no_finite_pairs', 'different_shapes'])
def test_invalid_surface_error_data_is_rejected(invalid):
    assert hasattr(gallery, 'surface_error_rows'), '顶面云图缺少逐时刻误差计算'
    frames = [] if invalid == 'empty' else [
        (5., np.array([np.nan]) if invalid == 'no_finite_pairs' else np.ones((2, 1)),
         np.array([20.]) if invalid == 'no_finite_pairs' else np.ones(2), [], '实验温度')]
    with pytest.raises(ValueError):
        gallery.surface_error_rows(frames)


def test_surface_error_curve_exports_real_png_pdf_and_unrounded_csv(project, surface_frames):
    import csv
    assert hasattr(gallery, 'save_surface_error_curve'), '缺少顶面误差曲线出图'
    out = complete_run(project)
    rows = gallery.save_surface_error_curve(out, 169., surface_frames)
    folder = out / '169.000W'
    with Image.open(folder / '顶面温度误差随时间.png') as image:
        assert image.width >= 2000 and image.height >= 1000
        assert image.info['dpi'][0] == pytest.approx(300., abs=.02)
        image.verify()
    assert (folder / '顶面温度误差随时间.pdf').read_bytes().startswith(b'%PDF')
    with (folder / '顶面温度误差逐时刻.csv').open(encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 2
    assert float(records[0]['时间_s']) == 5.
    assert float(records[0]['均方根误差_℃']) == rows[0]['均方根误差_℃']
    assert float(records[1]['平均偏差_℃']) == -.5
    assert int(records[1]['有效点数']) == 2


def test_surface_error_curves_are_visible_near_the_top_and_not_duplicated(project, surface_frames):
    assert hasattr(gallery, 'save_surface_error_curve'), '缺少顶面误差曲线出图'
    out = complete_run(project)
    gallery.save_surface_error_curve(out, 169., surface_frames)
    gallery.write_gallery(out, metrics(3.))
    fixed = gallery.publish_latest_gallery(project, out)
    text = fixed.read_text(encoding='utf-8')
    assert text.count('<h2>顶面温度误差随时间变化</h2>') == 1
    assert text.index('<h2>顶面温度误差随时间变化</h2>') < text.index('<h2>损失函数变化</h2>')
    assert_assets_from(fixed, out, embedded=True, expected_count=5)


def test_first_publication_updates_metrics_and_assets_and_preserves_original(project):
    fixed = project / FIXED
    fixed.parent.mkdir(parents=True)
    original = b'<!doctype html><html>historical original gallery</html>'
    fixed.write_bytes(original)
    out = complete_run(project)
    before = file_hashes(out)
    result = gallery.publish_latest_gallery(project, out)
    assert result == fixed
    assert (fixed.parent / '结果总览_原始运行.html').read_bytes() == original
    text = fixed.read_text(encoding='utf-8')
    assert out.name in text and '8000' in text and '验证最佳模型.pt' in text
    assert '<td>3.000</td>' in text
    assert_assets_from(fixed, out, embedded=True)
    assert file_hashes(out) == before
    record = json.loads((fixed.parent / '固定图册更新记录.json').read_text(encoding='utf-8'))
    assert record['来源运行目录'] == out.relative_to(project).as_posix()
    assert record['完成轮数'] == 8000
    assert record['检查点SHA256'] == json.loads((out / '测试出图锁定记录.json').read_text())['检查点SHA256']


def test_next_publication_changes_source_but_not_original_backup_or_runs(project):
    fixed = project / FIXED
    fixed.parent.mkdir(parents=True)
    fixed.write_text('original', encoding='utf-8')
    first = complete_run(project, '联合训练8000轮_第一次', error=3.)
    second = complete_run(project, '联合训练8000轮_第二次', error=7.)
    before = {out.name: file_hashes(out) for out in (first, second)}
    gallery.publish_latest_gallery(project, first)
    gallery.publish_latest_gallery(project, second)
    assert (fixed.parent / '结果总览_原始运行.html').read_text() == 'original'
    text = fixed.read_text(encoding='utf-8')
    assert '<td>7.000</td>' in text and '<td>3.000</td>' not in text
    assert_assets_from(fixed, second, embedded=True)
    assert {out.name: file_hashes(out) for out in (first, second)} == before


@pytest.mark.parametrize('invalid', ['missing_lock', 'unfinished', 'wrong_hash',
                                    'missing_checkpoint', 'missing_gallery', 'missing_metrics', 'invalid_metrics'])
def test_invalid_run_does_not_replace_fixed_page_or_create_backup(project, invalid):
    fixed = project / FIXED
    fixed.parent.mkdir(parents=True)
    fixed.write_text('old fixed gallery', encoding='utf-8')
    out = complete_run(project)
    lock_path = out / '测试出图锁定记录.json'
    if invalid == 'missing_lock':
        lock_path.unlink()
    elif invalid in ('unfinished', 'wrong_hash'):
        lock = json.loads(lock_path.read_text())
        lock['完成轮数' if invalid == 'unfinished' else '检查点SHA256'] = 7999 if invalid == 'unfinished' else '0' * 64
        lock_path.write_text(json.dumps(lock), encoding='utf-8')
    elif invalid == 'missing_checkpoint':
        (out / '验证最佳模型.pt').unlink()
    elif invalid == 'missing_gallery':
        (out / '结果总览.html').unlink()
    elif invalid == 'missing_metrics':
        (out / '测试指标_原始精度.json').unlink()
    else:
        (out / '测试指标_原始精度.json').write_text('{', encoding='utf-8')
    before = file_hashes(fixed.parent)
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        gallery.publish_latest_gallery(project, out)
    assert file_hashes(fixed.parent) == before


def test_publication_creates_fixed_directory_if_not_yet_present(project):
    out = complete_run(project)
    fixed = gallery.publish_latest_gallery(project, out)
    assert fixed == project / FIXED
    assert_assets_from(fixed, out, embedded=True)
    assert not (fixed.parent / '结果总览_原始运行.html').exists()


def test_source_outside_selected_project_is_rejected_without_changes(project):
    root = project / '单元测试项目'
    root.mkdir()
    out = complete_run(project)
    with pytest.raises(ValueError, match='项目'):
        gallery.publish_latest_gallery(root, out)
    assert not (root / FIXED).exists()


def test_original_run_can_publish_itself_without_losing_local_assets(project):
    out = complete_run(project, FIXED.parent.name)
    fixed = project / FIXED
    original = fixed.read_bytes()
    unchanged = {name: digest for name, digest in file_hashes(out).items() if name != '结果总览.html'}
    gallery.publish_latest_gallery(project, out)
    assert_assets_from(fixed, out, embedded=True)
    assert (out / '结果总览_原始运行.html').read_bytes() == original
    after = file_hashes(out)
    assert all(after[name] == digest for name, digest in unchanged.items())


@pytest.mark.parametrize('source_kind', ['new_run', 'canonical', 'canonical_bad_hash'])
def test_completed_export_automatically_publishes_the_fixed_gallery(project, source_kind):
    import polars as pl
    configs = project / 'configs'
    configs.mkdir()
    for name in ('data_metadata.yaml', 'splits.yaml'):
        shutil.copy2(ROOT / 'configs' / name, configs / name)
    splits = fixed_splits(project, 'all_training')
    processed = project / 'data/processed'
    processed.mkdir(parents=True)
    top = [{'split': 'test', 'source_dataset': 'test', 'power_w': power,
            'time_s': 5., 'r_m': radius, 'temperature_mean_k': 300., 'frame_weight': 1.}
           for power in splits['high']['test'] for radius in (0., .01)]
    sensors = [{'split': 'test', 'source_dataset': 'test', 'power_w': power,
                'sensor_type': tag, 'time_raw': time, 'radius_raw': .03, 'value_mean_raw': 25.}
               for power in splits['high']['test'] for tag in ('hot', 'cold') for time in (1., 5.)]
    pl.DataFrame(top).write_parquet(processed / 'test_ir_radial.parquet')
    pl.DataFrame(sensors).write_parquet(processed / 'test_sensor_ring_raw.parquet')
    settings = PhysicalSettings(295.15, 298.15, 295.15, 295.15, 295.15, .8, .02, 9., 4.3,
                                .5, .5, True, 8900., 400., 401., 3170., 700., 120., 7e-5)
    model = JointDeepONet(Geometry(time_max=5.), settings,
                         {'time_response': 'monotone_heating', 'width': 8, 'latent_dim': 8,
                          'blocks': 1, 'correction_width': 8, 'correction_depth': 1})
    config = {'visualization': {'raw_test_surface': '单元测试无原始像素', 'image_dpi': 30,
                                'animation_time_step_s': 5., 'animation_fps': 5}}
    out = complete_run(project, FIXED.parent.name if source_kind.startswith('canonical') else '联合训练8000轮_自动发布')
    original = (out / '结果总览.html').read_bytes()
    if source_kind == 'canonical_bad_hash':
        lock_path = out / '测试出图锁定记录.json'
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        lock['检查点SHA256'] = '0' * 64
        lock_path.write_text(json.dumps(lock), encoding='utf-8')
        before = file_hashes(out)
        with pytest.raises(RuntimeError, match='检查点|SHA256'):
            gallery.export_results(model, project, out, splits, config)
        assert file_hashes(out) == before
        return
    gallery.export_results(model, project, out, splits, config)
    fixed = project / FIXED
    assert fixed.is_file()
    sources = Images(fixed.read_text(encoding='utf-8')).sources
    assert len(sources) > 4
    assert_assets_from(fixed, out, embedded=True, expected_count=None)
    for power in splits['high']['test']:
        folder = out / f'{power:.3f}W'
        assert (folder / '顶面温度误差随时间.png').is_file()
        assert (folder / '顶面温度误差逐时刻.csv').is_file()
    record = json.loads((fixed.parent / '固定图册更新记录.json').read_text(encoding='utf-8'))
    assert record['来源运行目录'] == out.relative_to(project).as_posix()
    assert (out / '结果总览.html').is_file()
    if source_kind == 'canonical':
        assert (out / '结果总览_原始运行.html').read_bytes() == original
        assert record['来源图册文件名'] == '结果总览_本次重出图.html'
        assert (out / record['来源图册文件名']).is_file()
