import numpy as np
import subprocess


class ProteinState:
    def __init__(self,
                 n_residues: int = 20,
                 angles: np.ndarray = None):
        if angles is None:
            self._angles = np.array([[180.0 for _ in range(n_residues)] for _ in range(2)])
        else:
            self._angles = angles
            self._angles %= 360

        assert self._angles.shape[0] == 2

        self._hash = None

    def __str__(self):
        return " ".join([" ".join([str(a) for a in phi_psi]) for phi_psi in self._angles.T])

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.__str__())

    def __eq__(self, other):
        return np.all(self._angles == other._angles)

    def __ne__(self, other):
        return not self.__eq__(other)

    def angles(self) -> np.ndarray:
        return self._angles

    def n_residues(self) -> int:
        return self._angles.shape[1]

    def eval_state(self) -> float:
        angles_file = "angles_temp.txt"
        with open(angles_file, "w") as f:
            f.write(f"{self}\n")

        output = subprocess.check_output(["redcraft", "molan", "-e", "-d" "RDC_new", "-p", ".", "-m", "2", angles_file])

        score = float(output.decode("utf-8").split()[-1])
        return score

    def do_action(self, action: np.ndarray):
        action = action.reshape(self._angles.shape)
        assert self._angles.shape == action.shape
        new_angles = self._angles + action
        new_angles %= 360
        return ProteinState(angles=new_angles)

    def l2_norm(self, other) -> float:
        return np.sqrt(((self._angles - other._angles) ** 2).mean(axis=None))
