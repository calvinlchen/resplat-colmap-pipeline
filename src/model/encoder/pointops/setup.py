import os
import subprocess
from setuptools import setup
from distutils.sysconfig import get_config_vars


def use_vs2022_on_windows():
    if os.name != "nt" or os.environ.get("VSCMD_ARG_TGT_ARCH"):
        return

    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    vcvars = next((path for path in candidates if os.path.exists(path)), None)
    if vcvars is None:
        return

    result = subprocess.run(
        f'cmd.exe /d /c "call "{vcvars}" >nul && set"',
        check=True,
        capture_output=True,
        text=True,
        shell=True,
    )
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            os.environ[key] = value

    sdk_bin = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
    if os.path.exists(os.path.join(sdk_bin, "rc.exe")):
        os.environ["PATH"] = sdk_bin + os.pathsep + os.environ.get("PATH", "")

    os.environ.setdefault("DISTUTILS_USE_SDK", "1")


use_vs2022_on_windows()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

(opt,) = get_config_vars("OPT")
opt = opt or ""
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

cxx_args = ["/Z7"] if os.name == "nt" else ["-g"]
link_args = ["/MANIFEST:NO"] if os.name == "nt" else []

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={"cxx": cxx_args, "nvcc": ["-O2"]},
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
