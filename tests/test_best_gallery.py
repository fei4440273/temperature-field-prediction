"""The fixed overview shows only the retained model's development results."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / '研究记录/联合训练600轮_第09轮_低保真初温平滑锚定_20260920'
OVERVIEW = ROOT / '研究记录/联合训练8000轮_20260917_114704/结果总览.html'


class GalleryAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.images.append(dict(attrs)['src'])
        if tag == 'a':
            self.links.append(dict(attrs)['href'])


def test_fixed_overview_only_shows_retained_model_and_readable_development_images():
    assert OVERVIEW.is_file()
    assert OVERVIEW.stat().st_size < 250_000  # Reject the old embedded test-result gallery before reading it.
    html = OVERVIEW.read_text(encoding='utf-8')
    assert '第09轮' in html and '训练与验证' in html
    assert all(metric in html for metric in ('3.2686', '1.0722', '0.6504'))
    assert all(power in html for power in ('115.200W', '403.000W', '630.500W'))
    assert all(power not in html for power in ('169.000W', '339.000W', '634.000W'))

    page = GalleryAssets()
    page.feed(html)
    assert len(page.images) == 16  # Validation loss plus five plots for each of three powers.
    assert '训练功率逐时图' in html
    powers = yaml.safe_load((ROOT / 'configs/splits.yaml').read_text(encoding='utf-8'))
    for power in powers['high_fidelity']['training_powers_w']:
        assert sum(f'/训练/{power:.3f}W/' in link for link in page.links) == 5
    assert '[模型](' not in html
    assert any(link.endswith('/验证最佳模型.pt') for link in page.links)
    for source in page.links:
        parsed = urlsplit(source)
        assert not parsed.scheme and not parsed.netloc
        assert (OVERVIEW.parent / unquote(parsed.path)).is_file() or source.startswith('#')
    for source in page.images:
        parsed = urlsplit(source)
        assert not parsed.scheme and not parsed.netloc
        image = (OVERVIEW.parent / unquote(parsed.path)).resolve()
        assert BEST in image.parents and image.is_file()
        with Image.open(image) as visual:
            visual.verify()
