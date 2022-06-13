from importlib.abc import Loader
import yaml
import numpy as np


Q = np.array([[1,0.2],[0.2,0.5]])
R = np.array([0.75])


dict_covar = [{"state covar": np.ndarray.tolist(Q) , "senser covar": float(R[0])}]
with open(r'test.yaml','w') as file:
    yaml.dump(dict_covar, file)



with open(r'read.yaml') as file:
    covar = yaml.load(file, Loader = yaml.FullLoader)

Q_mat = np.array(covar[0]['state covar'])
R_mat = np.array(covar[0]['senser covar'])
print(Q_mat)
print(R_mat)