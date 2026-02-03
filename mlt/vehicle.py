from dataclasses import dataclass
import pinocchio as pin
import numpy as np

@dataclass
class VehicleProperties:
    # Suspension
    s_k: float = 0
    s_c: float = 0

@dataclass
class SetupProperties:
    pass


class Vehicle:
    def __init__(self):
        self.model = pin.buildModelsFromUrdf("vehicle.urdf", root_joint=pin.JointModelFreeFlyer())
        
if __name__ == "__main__":
    v = Vehicle()
    foo, *x = v.model

    print(foo.njoints)
    f_ext = [pin.Force.Zero() for _ in range(foo.njoints)]

    
    f_ext[6] = pin.Force(np.array([0, 0, -100]), np.array([0, 0, 0]))
    data = foo.createData()

    qddot = pin.aba(foo, data, pin.neutral(foo) , np.zeros(foo.nv), np.zeros(foo.nv), f_ext)

    for i in range(foo.njoints):
        print(f"Index {i}: {foo.names[i]}, mass={foo.inertias[i].mass:.4f}")
    print(qddot, type(qddot))

    for i in range(foo.njoints):
        name = foo.names[i]
        pos = data.oMi[i].translation
        rot = data.oMi[i].rotation
        
        print(f"Joint {i} ({name}):")
        print(f"  Position: {pos}")
        print(f"  Rotation:\n{rot}\n")
    # print(data.v.linear, data.v.angular)
    # print(data.a.linear, data.a.angular)

    pin.computeAllTerms(foo, data, np.zeros(foo.nq), np.zeros(foo.nv),)
    print(data.M)


    