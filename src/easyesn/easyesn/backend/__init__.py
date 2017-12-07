import sys
import os
import json

#default backend
_BACKEND = 'numpy'

#inspired by Keras backend handling

# Obtain easyesn base directory path: either ~/.easyesn or /tmp.
_easyesn_base_dir = os.path.expanduser('~')
if not os.access(_easyesn_base_dir, os.W_OK):
    _easyesn_base_dir = '/tmp'
_easyesn_dir = os.path.join(_easyesn_base_dir, '.easyesn')

# Attempt to read easyesn config file.
_config_path = os.path.expanduser(os.path.join(_easyesn_dir, 'easyesn.json'))
if os.path.exists(_config_path):
    try:
        _config = json.load(open(_config_path))
    except ValueError:
        _config = {}
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'numpy', 'np', 'cupy', 'cp'}
    
    if _backend == "cp":
        _backend = "cupy"
    if _backend == "np":
        _backend = "numpy"

    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_easyesn_dir):
    try:
        os.makedirs(_easyesn_dir)
    except OSError:
        pass

if not os.path.exists(_config_path):
    _config = {
        'backend': _BACKEND
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on EASYESN_BACKEND flag, if applicable.
if 'EASYESN_BACKEND' in os.environ:
    _backend = os.environ['EASYESN_BACKEND']
    assert _backend in {'numpy', 'np', 'cupy', 'cp'}
    
    if _backend == "cp":
        _backend = "cupy"
    if _backend == "np":
        _backend = "numpy"

    _BACKEND = _backend

if _BACKEND == 'cupy':
    sys.stderr.write('Using CuPy backend\n')
    from .cupyBackend import *
elif _BACKEND == 'numpy':
    sys.stderr.write('Using Numpy backend.\n')
    from .numpyBackend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))

def backendName():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend which is currently used.
    """
    return _BACKEND