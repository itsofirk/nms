import os

if os.path.normpath(os.getcwd()) != os.path.normpath(os.path.dirname(__file__)):
    raise RuntimeError(f"{os.path.basename(__file__)} can only be executed from within the same directory")

if __name__ == '__main__':
    mock = False
    if mock:
        from service import mock as main
    else:
        from service import test as main

    main()
