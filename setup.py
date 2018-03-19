from setuptools import setup
setup(name='pymam',
      version='1.0',
      packages = ['pymam'],
      install_requires=['music21','sounddevice','pygame','librosa']
      )

# Doc generation
# sphinx-apidoc . --full -o doc -H 'pymam' -A 'Pablo E. Riera' -V '1.0'
