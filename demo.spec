# demo.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

a = Analysis(
    ['app/demo.py'],          # main script
    pathex=[],
    binaries=[],
    datas=[
        ('app', 'app'),
        ('data', 'data'),
        ('poppler_dir', 'poppler_dir'),
    ] + collect_data_files('safehttpx') + collect_data_files('groovy') + collect_data_files('gradio', include_py_files=True),
    hiddenimports=collect_submodules('safehttpx') + collect_submodules('groovy') + collect_submodules('gradio'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='demo'
)
