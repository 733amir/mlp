class Utils:

    _debug = True
    _verbosity = 1

    @classmethod
    def log(cls, title, data, verbosity=1):
        if cls._debug and cls._verbosity >= verbosity:
            print(title, *data, sep='\n', end='\n\n')
