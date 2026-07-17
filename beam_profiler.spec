# beam_profiler.spec
#
# Build with:  pyinstaller beam_profiler.spec
#
# This is written for --onedir (recommended for SciPy/OpenCV/matplotlib apps --
# see the packaging notes: --onefile unpacks everything to a temp folder on
# every launch, which is slow and unnecessary for a lab desktop app).
#
# Before building:
#   - Put GUI1.ui next to this .spec file (or fix the path in `datas` below).
#   - Put your icon as app_icon.ico next to this .spec file.
#   - pip install pyinstaller  (in the same environment/venv as everything else)

import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = [
    ('GUI1.ui', '.'),        # loaded at runtime via uic.loadUi -- PyInstaller's
                             # static analysis can't see this, so it must be
                             # listed explicitly or the frozen exe crashes on
                             # the very first line of MainWindow.__init__.
    ('beamappicon.ico', '.'),   # only needed if you load this at runtime (e.g. for
                             # setWindowIcon); the .exe's own icon is separate,
                             # set below via `icon=`, not from this list.
]
datas += collect_data_files('matplotlib')  # mpl-data: fonts, style files, etc.

a = Analysis(
    ['main1.py'],          # <-- change to your actual main script filename
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # pylablib's hardware backends are imported dynamically in try/except
        # blocks in this file, which can make PyInstaller's static import
        # scanner miss them entirely -- if a packaged build says "Thorlabs
        # Scientific Camera SDK not available" even though it works fine when
        # run with `python`, this is almost always why.
        'pylablib.devices.Thorlabs',
        'pylablib.devices.uc480',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VOILALabBeam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # equivalent to --windowed: no console window
    icon='beamappicon.ico',     # stamps the icon onto the .exe file itself
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VOILALabBeam',     # this is the folder name under dist/
)
