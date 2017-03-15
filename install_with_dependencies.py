import pip

def install(package):
	pip.main(['install',package ])

def upgrade(package):
	pip.main(['install',"--upgrade",package ])

if __name__ == '__main__':
    install("music21")
    install("sounddevice")
    upgrade("https://github.com/pabloriera/pymam/tarball/master")
    