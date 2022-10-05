from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import shutil
import subprocess

try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True
import pointnet2

# cuda_version = 'cuda-11.2'
cuda_version = 'cuda-11.3'

def get_pkg_config_include_lib_paths(lib_name=''):
    # cmd = 'pkg-config cuda-11.3 --cflags'
    cmd = 'pkg-config ' + lib_name + ' --cflags --libs'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    p_status = p.wait()

    if err is None:
        return output.decode('utf-8'), p_status
    else:
        return output.decode('utf-8'), err, p_status


# _ext_src_root = "pointnet2/_ext-src/"   # original
_ext_src_root = "pointnet2/_ext-src"     # BERNARDO
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

out_cuda = get_pkg_config_include_lib_paths(lib_name=cuda_version)[0]

# requirements = ["etw_pytorch_utils==1.1.1", "h5py", "pprint", "enum34", "future"]  # original
requirements = ["etw_pytorch_utils==1.1.0", "h5py", "enum34", "future"]              # BERNARDO

setup(
    name="pointnet2",
    version=pointnet2.__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
		# "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],    # original
		# "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],   # original
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root)), out_cuda],      # BERNARDO
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root)), out_cuda],     # BERNARDO
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

_ext_name = glob.glob("pointnet2/*.so")
shutil.copy(_ext_name[0], '.')
