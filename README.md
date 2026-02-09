# polycrystal

This is a package with microstructural models for polycrystalline materials.  It contains subpackages for crystal orientations, microstructure (in the sense of grain/phase assignment), linear anisotropic elasticty and slip modeling. A primary intent has been to use this package to provide material properties for finite element simulations using the companion package, _polycrystalx_.

There is some overlap with HEXRD in representation of rotation matrices and crystal symmetries.

# Installation
You can use `pip` to install polycrystal. After downloading the polycrystal package, go to the directory above and run:
```
pip install -r polycrystal/requirements.txt
pip install -e polycrystal
```
That assumes you downloaded it to a directory named `polycrystal`.
