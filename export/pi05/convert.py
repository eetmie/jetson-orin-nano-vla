"""Convert the pinned upstream checkpoint independently of its directory name."""
from pathlib import Path
import sys


def patch_selector(source):
    # The pinned converter receives pi05 but selects adaptive norms by path.
    needle = 'if "pi05" in checkpoint_dir:'
    if source.count(needle) != 3:
        raise RuntimeError('Pinned converter normalization selector changed')
    return source.replace(needle, 'if pi05:')


def main():
    from openpi_model import register_configs
    register_configs()
    import torch
    # Seed the unused language-output projection for repeatable file contents.
    torch.manual_seed(0)
    script = '/opt/openpi/examples/convert_jax_model_to_pytorch.py'
    sys.argv[0] = script
    source = patch_selector(Path(script).read_text())
    exec(compile(source, script, 'exec'), {'__name__': '__main__', '__file__': script})


if __name__ == '__main__':
    main()
