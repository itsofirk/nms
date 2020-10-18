import os

if os.getcwd() != os.path.dirname(__file__):
    raise ModuleNotFoundError(f"{os.path.basename(__file__)} can only run from within the same directory")

if __name__ == '__main__':
    mock = False
    if mock:
        from service import mock as main
    else:
        from service import main

    main()
