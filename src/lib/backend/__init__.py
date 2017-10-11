_BACKEND = 'numpy'

if _BACKEND == 'cupy':
    sys.stderr.write('Using CuPy backend\n')
    from cupyBackend import *
elif _BACKEND == 'numpy':
    sys.stderr.write('Using Numpy backend.\n')
    from numpyBackend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend which is currently used.
    """
    return _BACKEND