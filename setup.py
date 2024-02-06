from setuptools import find_packages, setup
from typing import List

HYPHEN_e_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:


    """
    This function will return the list of libraries in requirements.txt file
    """

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[
            req.replace('\n', "") for req in requirements
            ]
        # Or you can use this line as well
        # requirements = [
        #     req.strip() for req in file_obj.readlines()  # Remove leading/trailing whitespace
        # ]
        if HYPHEN_e_DOT in requirements:
            requirements.remove(HYPHEN_e_DOT)
            
    return requirements




setup(
    name="Energy Efficiency Model",
    version="0.0.1",
    author="Hasham Akram",
    author_email="hashamakram50@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)