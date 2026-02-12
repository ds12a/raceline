from dataclasses import dataclass

from track_import.track import Track
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import yaml
from mlt.pacejka import AWSIMPacejka
from itertools import product


@dataclass
class VehicleProperties:

    # Mass properties
    m_sprung: float  # Sprung mass
    m_unsprung: float  # Unsprung mass

    # Inertia properties
    i_xx: float  # Roll inertia
    i_yy: float  # Pitch inertia
    i_zz: float  # Yaw inertia

    # Geometry properties
    g_com_h: float  # COM height
    g_a: list  # Axles to COM [front, rear]
    g_t1: list  # Track widths [front, rear]
    g_S: float  # Frontal area
    g_hq1: list  # Roll centers [front, rear] (guess)

    # Aero properties
    a_Cx: 0.8581  # Drag coeff
    a_Cz: list  # Downforce coeff [front, rear]

    # Suspension
    s_k: list  # Spring stiffness [[FL, FR], [RL, RR]]
    s_c: list  # Damping coeff [[FL, FR], [RL, RR]]

    # Tire
    t_rw: list  # Tire radius [front, rear]
    t_Dy1: list
    t_Dy2: list
    t_Cy: list
    t_sypeak: list
    t_Fznom: list
    t_Ey: list
    # Setup
    p_kb: 0.5  # Brake Bias
    p_karb: list  # ARB stiffness [front, rear]

    @staticmethod
    def load_yaml(config):
        with open(config, "r") as f:
            all_things = yaml.safe_load(f)

        return VehicleProperties(**all_things)


@dataclass
class EnvProperties:
    rho: 1.1839  # kg / m3


class Vehicle:
    def __init__(self, config, track: Track, opti: ca.Opti):

        # Loading vehicle configuration properties
        self.prop = VehicleProperties.load_yaml(config)
        self.env = EnvProperties(1.1839)
        self.track = track
        self.pacejka = [
            AWSIMPacejka(
                self.prop.t_sypeak[i],
                self.prop.t_Cy[i],
                self.prop.t_Dy1[i],
                self.prop.t_Dy2[i],
                self.prop.t_Fznom[i],
                self.prop.t_Ey[i],
            )
            for i in range(2)
        ]

        # Initializing numeric model
        self.model = pin.Model()
        self._init_pin_tree()

        # Initializes symbolic CasADi modules
        self.opti = opti
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Initializes CasADi functions
        self._init_w3_func()
        self._init_w6_func()

    def _init_pin_tree(self):
        """
        Initializes the pinocchio vehicle model
        """

        # Floating track joint
        self.track_id = self.model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            "track",
        )
        # Prismatic defining vehicle position on track
        self.road_lat_id = self.model.addJoint(
            self.track_id,
            pin.JointModelPY(),
            pin.SE3.Identity(),
            "road_lat",
        )
        # Revolute defining vehicle yaw
        self.yaw_id = self.model.addJoint(
            self.road_lat_id,
            pin.JointModelRZ(),
            pin.SE3.Identity(),
            "yaw",
        )
        # Prismatic defining suspension vertical travel
        self.vert_id = self.model.addJoint(
            self.yaw_id,
            pin.JointModelPZ(),
            pin.SE3.Identity(),
            "vert",
        )
        # Revolute defining vehicle pitch
        self.pitch_id = self.model.addJoint(
            self.vert_id,
            pin.JointModelRY(),
            pin.SE3.Identity(),
            "pitch",
        )
        # Revolute defining vehicle roll
        self.roll_id = self.model.addJoint(
            self.pitch_id,
            pin.JointModelRX(),
            pin.SE3.Identity(),
            "roll",
        )

        # Defines inertial/mass values
        self.unsprung = pin.Inertia(self.prop.m_unsprung, np.zeros(3), np.zeros((3, 3)))
        self.sprung = pin.Inertia(
            self.prop.m_sprung,
            np.zeros(3),
            np.diag(
                [
                    self.prop.i_xx,
                    self.prop.i_yy,
                    self.prop.i_zz,
                ]
            ),
        )
        self.model.appendBodyToJoint(self.yaw_id, self.unsprung, pin.SE3.Identity())
        self.model.appendBodyToJoint(
            self.roll_id,
            self.sprung,
            pin.SE3(np.eye(3), np.array([0, 0, self.prop.g_com_h])),
        )

    def _init_w6_func(self):
        """
        Generates W66E, the external aerodynamic wrench.
        """

        v_3 = ca.MX.sym("v_3", 6)

        w6 = (
            -self.env.rho
            / 2
            * self.prop.g_S
            * (v_3[0]) ** 2
            * ca.vertcat(
                self.prop.a_Cx,
                self.prop.a_Cz[0] + self.prop.a_Cz[1],
                self.prop.a_Cz[1] * self.prop.g_a[1] - self.prop.a_Cz[1] * self.prop.g_a[1],
            )
        )

        self.w6_func = ca.Function("W6", [v_3], [w6])

    def _init_w3_func(self):
        """
        Initializes W33E, the external tire force wrench.
        """

        u = ca.SX.sym("u", 3)  # Control [f_xa, f_xb, delta]
        v_3 = ca.SX.sym("v_3", 6)  # Twist vector (frame 3)
        f_z = ca.SX.sym("f_z", 2, 2)  # z forces on each wheel
        f_xa, f_xb, delta = ca.vertsplit(u)
        v_3x, v_3y, _, _, _, omega_3z = ca.vertsplit(v_3)

        # Defining tire slip alpha
        alpha_out = ca.vertcat(
            delta - (v_3y + omega_3z * self.prop.g_a[0]) / v_3x,
            -(v_3y - omega_3z * self.prop.g_a[1]) / v_3x,
        )
        alpha = ca.Function("alpha", [v_3, u], [alpha_out])(v_3, u)

        # Defining Pacejka lateral force f_ijy
        f_y_out = ca.vertcat(
            *[
                ca.horzcat(*[self.pacejka[i](alpha[i], f_z[i, j]) for j in range(2)])
                for i in range(2)
            ]
        )
        f_y = ca.Function("f_y", [f_z, u, v_3], [f_y_out])(f_z, u, v_3)

        # Defining longitudinal force f_ijx
        f_x_out = ca.vertcat(
            ca.horzcat(*(2 * [f_xb * self.prop.p_kb / 2])),
            ca.horzcat(*(2 * [f_xb * (1 - self.prop.p_kb) + f_xa])) / 2,
        )
        f_x = ca.Function("f_x", [u], [f_x_out])(u)

        # Defines expressions X1, X2, Y1, Y2
        self.X1_func = ca.Function(
            "X1",
            [f_z, u, v_3],
            [(f_x[0, 0] + f_x[0, 1]) * ca.cos(delta) - (f_y[0, 0] + f_y[0, 1]) * ca.sin(delta)],
        )
        self.X2_func = ca.Function("X2", [f_z, u, v_3], [f_x[1, 0] + f_x[1, 1]])
        self.Y1_func = ca.Function(
            "Y1",
            [f_z, u, v_3],
            [(f_y[0, 0] + f_y[0, 1]) * ca.cos(delta) + (f_x[0, 0] + f_x[0, 1]) * ca.sin(delta)],
        )
        self.Y2_func = ca.Function("Y2", [f_z, u, v_3], [f_y[1, 0] + f_y[1, 1]])

        # Define W3e
        X1 = self.X1_func(f_z, u, v_3)
        X2 = self.X2_func(f_z, u, v_3)
        Y1 = self.Y1_func(f_z, u, v_3)
        Y2 = self.Y2_func(f_z, u, v_3)
        w3e = ca.vertcat(X1 + X2, Y1 + Y2, Y1 * self.prop.g_a[0] - Y2 * self.prop.g_a[1])
        self.w3e_func = ca.Function("W3e", [f_z, u, v_3], [w3e])

    def _init_fz_func(self):
        """
        Generates fz, the downforce calculated based on vehicle weight shifts and aerodynamic loads
        """
        u = ca.SX.sym("u", 3)  # Control [f_xa, f_xb, delta]
        v_3 = ca.SX.sym("v_3", 6)  # Twist vector (frame 3)
        f_z = ca.SX.sym("f_z", 2, 2)  # z forces on each wheel

        f_za = ca.SX.sym("f_za") # Aerodynamic downforce TODO pass in or recalculate

        f_xa, f_xb, delta = ca.vertsplit(u)
        v_3x, v_3y, _, _, _, omega_3z = ca.vertsplit(v_3)
        f_1z, f_2z = ca.vertsplit(f_z)
        f_11z, f_12z = ca.horzsplit(f_1z)
        f_21z, f_22z = ca.horzsplit(f_2z)

        # Static load

        # Aero downforce

        # Longitudinal shift
        
        # Lateral shift


    def rnea(
        self,
        q_1: float,
        q_1_dot: ca.SX,
        q_1_ddot: ca.SX,
        q: ca.SX,
        q_dot: ca.SX,
        q_ddot: ca.SX,
        f_ext: list[cpin.Force],
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """
        Performs RNEA

        Args:
            q_1 (float): Normalized parameter [0, 1] indicating position along track
            q_1_dot (ca.SX): First time derivative of q_1
            q_1_ddot (ca.SX): Second time derivative of q_1
            q (ca.SX): Joint positions excluding free floating joint
            q_dot (ca.SX): First time derivatives of joint positions, excluding free floating joint
            q_ddot (ca.SX): Second time derivatives of joint positions, excluding free floating joint
            f_ext (list[cpin.Force]): External forces applied

        Returns:
            tuple[np.ndarray, tuple[float, float, float]]: Torques (τ1, ..., τ6) and structural wrench components (f3z, m3x, m3y)
        """

        # Calculates track position, vel, accel
        track_q = self.track.state(np.array([q_1]) * self.track.length)[0]
        track_v_spacial = self.track.der_state(np.array([q_1]) * self.track.length, n=1)[0]
        track_a_spacial = self.track.der_state(np.array([q_1]) * self.track.length, n=2)[0]

        # Convert from [0, 1] normalized parameter to arc length to time derivative
        track_v = track_v_spacial * self.track.length * q_1_dot
        track_a = (
            track_a_spacial * self.track.length**2 * q_1_dot**2
            + track_v_spacial * self.track.length * q_1_ddot
        )

        # Compute rotation matrix from body to world
        R = pin.rpy.rpyToMatrix(*track_q[3:6][::-1])  # We store in zyx (yaw, pitch, roll)

        # Create full state
        q = np.hstack([track_q[:3], pin.Quaternion(R).coeffs(), q])

        # Calculates v (track velocity) and a (track accel)
        theta, mu, phi = track_q[3:6]
        theta_dot, mu_dot, phi_dot = track_v[3:6]

        # Precompute because we like efficiency
        c_mu = np.cos(mu)
        s_mu = np.sin(mu)
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)

        # Rotation Jacobians
        J_e = np.array([[0, -s_theta, c_theta * c_mu], [0, c_theta, s_theta * c_mu], [1, 0, -s_mu]])
        J_e_dot = np.array(
            [
                [0, -theta_dot * c_theta, -theta_dot * s_theta * c_mu - mu_dot * c_theta * s_mu],
                [0, -theta_dot * s_theta, theta_dot * c_theta * c_mu - mu_dot * s_theta * s_mu],
                [0, 0, -mu_dot * c_mu],
            ]
        )

        # Compute ω and concatenate with spacial velocities and given joint velocities
        # Rotate from world to body
        body_linear_v = R.T @ track_v[:3]
        body_angular_v = R.T @ J_e @ track_v[3:6]

        v = np.hstack([body_linear_v, body_angular_v, q_dot])

        # Compute dω and concatentate with accelerations and given joint accelerations
        # Rotate from world to body
        a = np.hstack(
            [
                R.T @ track_a[:3] - np.cross(body_angular_v, body_linear_v),
                R.T @ (J_e @ track_a[3:6] + J_e_dot @ track_v[3:6]),
                q_ddot,
            ]
        )

        torques = cpin.rnea(self.model, self.data, q, v, a, f_ext)

        return torques, (
            self.data.f[3].linear[3],
            self.data.f[3].angular[0],
            self.data.f[3].angular[1],
        )

    def set_constraints(self, q_1, q_1_dot, q_1_ddot, q, dq, ddq, f_z, u, v_3):
        v_3x, v_3y, v_3z = ca.vertsplit(v_3)

        q_dot = dq * q_1_dot
        q_ddot = ddq * q_1_dot**2 + dq * q_1_ddot

        f_ext = [cpin.Force.Zero() for _ in range(self.model.njoints)]
        
        f_3x, f_3y, m_3z = ca.vertsplit(self.w3e_func(f_z, u, v_3))
        f_xa, f_za, m_ya = ca.vertsplit(self.w6_func(f_z, u, v_3))

        # TODO Check these indicies
        f_ext[3] = cpin.Force(np.array([f_3x, f3_y, 0]), np.array([0, 0, m_3z]))
        f_ext[6] = cpin.Force(np.array([f_xa, 0, f_za]), np.array([0, m_ya, 0]))
        

        torques, (f_3z, m_3x, m_3y) = self.rnea(q_1, q_1_dot, q_1_ddot, q, q_dot, d_ddot, f_ext)

        for i in range(3):
            self.opti.subject_to(torques[i] == 0)
        # self.opti.subject_to(torques[3] == )


if __name__ == "__main__":
    v = Vehicle("mlt/vehicle_properties/DallaraAV24.yaml", None, ca.Opti())
    v._init_w3_func()
    v.w3e_func.generate("foo")
    print(v.w3e_func(np.random.randn(2, 2), np.random.randn(3), np.random.randn(6)))
    # print(v.model)
    # print(v.model.nbodies)
    # foo = v.model

    # print(cpin.neutral(foo))
    # f_ext = [cpin.Force.Zero() for _ in range(foo.njoints)]

    # f_ext[6] = cpin.Force(np.array([0, 0, -1000]), np.array([0, 0, 0]))
    # data = foo.createData()

    # torques = cpin.rnea(foo, data, cpin.neutral(foo), np.zeros(foo.nv), np.zeros(foo.nv), f_ext)
    # print(torques)
    # print(data.f[3])

    # for i in range(foo.njoints):
    #     print(f"Index {i}: {foo.names[i]}, mass={foo.inertias[i].mass:.4f}")
    # print(qddot, type(qddot))

    # for i in range(foo.njoints):
    #     name = foo.names[i]
    #     pos = data.oMi[i].translation
    #     rot = data.oMi[i].rotation

    #     print(f"Joint {i} ({name}):")
    #     print(f"  Position: {pos}")
    #     print(f"  Rotation:\n{rot}\n")
    # # print(data.v.linear, data.v.angular)
    # # print(data.a.linear, data.a.angular)

    # pin.computeAllTerms(foo, data, np.zeros(foo.nq), np.zeros(foo.nv),)
    # print(data.M)
