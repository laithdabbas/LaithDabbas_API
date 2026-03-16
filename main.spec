# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None


a = Analysis(
    ['app\\main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('app', 'app'),
        ('data', 'data'),
        ('poppler_dir', 'poppler_dir'),
        ('spam_model.pkl', '.'),
    ]
    + collect_data_files('safehttpx')
    + collect_data_files('groovy')
    + collect_data_files('gradio', include_py_files=True),
    hiddenimports=
        collect_submodules('safehttpx')
        + collect_submodules('groovy')
        + collect_submodules('gradio'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='start_server',
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
    name='start_server',
)
