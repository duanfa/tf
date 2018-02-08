import platform


def getNutstorDir():
    os = platform.system()
    global NutstorDir
    if(os == "Linux"):
        NutstorDir = "/home/socket/Nutstore"
    else:
        NutstorDir = "/Users/duanfa/Documents/Nutstore"
    return NutstorDir
