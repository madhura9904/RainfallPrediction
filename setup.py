from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements


setup(
    name='RainfallPrediction',
    version='0.0.1',
    author='Madhura',
    author_email='madhura9904@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()

) 

#Calling get_requirements("requirements.txt") would return: ['numpy', 'pandas', 'scikit-learn']

#What Happens When You Run setup.py install?
#get_requirements("requirements.txt") runs and fetches all dependencies.
#The install_requires argument in setup() gets populated with those dependencies.

'''Why is this useful?
Avoids duplicating dependencies in setup.py and requirements.txt.
Ensures your package has the correct dependencies installed.
Simplifies package management and installation.'''