mock = True
if mock:
    from nms import mock as main
else:
    from nms import main

if __name__ == '__main__':
    main()
