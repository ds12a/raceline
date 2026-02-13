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
    g_t: list  # Track widths [front, rear]
    g_S: float  # Frontal area
    g_hq1: list  # Roll centers [front, rear] (guess)
    g_steer_max = 0.179008128  # Max steering angle

    # Aero properties
    a_Cx: 0.8581  # Drag coeff
    a_Cz: list  # Downforce coeff [front, rear]

    # Suspension
    s_k: list  # Spring stiffness [[FL, FR], [RL, RR]]
    s_c: list  # Damping coeff [[FL, FR], [RL, RR]]

    # Tire
    t_rw: list  # Tire radius [front, rear]
    t_Dx1: list
    t_Dx2: list
    t_Dy1: list
    t_Dy2: list
    t_Cy: list
    t_sypeak: list
    t_Fznom: list
    t_Ey: list

    # Engine
    e_max = 341_000  # Max engine power

    # Setup
    p_kb: 0.5  # Brake Bias
    p_karb: list  # ARB stiffness [front, rear]

    @staticmethod
    def load_yaml(config):
        with open(config, "r") as f:
            all_things = yaml.safe_load(f)

        return VehicleProperties(**all_things)

    def p_phi_1(self, p):
        return (p[0, 0] + p[0, 1]) / 4 * self.g_t[0] ** 2 + self.p_karb[0]

    def p_phi_2(self, p):
        return (p[1, 0] + p[1, 1]) / 4 * self.g_t[1] ** 2 + self.p_karb[1]

    def p_phi(self, p):
        return self.p_phi_1(p) + self.p_phi_2(p)

    def p_theta(self, p):
        (p[0, 0] + p[0, 1]) * self.g_a[0] ** 2 + (p[1, 0] + p[1, 1]) * self.g_a[1] ** 2

    def p(self, p):
        return p[0, 0] + p[0, 1] + p[1, 0] + p[1, 1]


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
        self.f_y_func = ca.Function("f_y", [f_z, u, v_3], [f_y_out])
        f_y = self.f_y_func(f_z, u, v_3)

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

        f_za = ca.SX.sym("f_za")  # Aerodynamic downforce

        # RNEA outputs for W_3J
        f_3z = ca.SX.sym("f_3z")  # Downforce
        m_3x = ca.SX.sym("m_3x")  # Roll
        m_3y = ca.SX.sym("m_3y")  # Pitch

        # W6
        m_ya = ca.SX.sym("m_ya")

        f_xa, f_xb, delta = ca.vertsplit(u)
        v_3x, v_3y, _, _, _, omega_3z = ca.vertsplit(v_3)
        f_1z, f_2z = ca.vertsplit(f_z)
        f_11z, f_12z = ca.horzsplit(f_1z)
        f_21z, f_22z = ca.horzsplit(f_2z)

        # Static load
        l = sum(self.prop.g_a)
        f_z0 = ca.Function(
            "f_zi0",
            [v_3, f_za, f_3z],
            [(f_3z - f_za) * ca.vertcat(*[(l - self.prop.g_a[i]) / (2 * l) for i in range(2)])],
        )

        # Aero downforce is passed in

        # Longitudinal shift
        delta_f_z = ca.Function("delta_f_z", [m_3y], [-(m_3y - m_ya) / (2 * l)])

        # Lateral shift
        Y = [self.Y1_func(f_z, u, v_3), self.Y2_func(f_z, u, v_3)]
        k_phi = [self.prop.p_phi_1(self.prop.s_k), self.prop.p_phi_2(self.prop.s_k)]
        lateral_delta_f_z_out = [
            ca.vertcat(
                *[
                    (k_phi[i] / sum(k_phi) * (-m_3x - sum(Y)) + self.prop.g_hq1[i] * Y[i])
                    / self.prop.t[i]
                    for i in range(2)
                ]
            )
        ]

        lateral_delta_f_z = ca.Function(
            "lateral_delta_f_z", [f_z, u, v_3, m_3x], lateral_delta_f_z_out
        )(f_z, u, v_3, m_3x)

        f_z0_out = f_z0(v_3, f_za, f_3z)
        lateral_delta_f_z_out = lateral_delta_f_z(f_z, u, v_3, m_3x)

        self.f_z_func = ca.Function(
            "f_z",
            [f_z, u, v_3, f_za, f_3z, m_3x, m_3y, m_ya],
            [
                ca.vertcat(
                    *[
                        ca.horzcat(
                            *[
                                f_z0_out[i]
                                + f_za
                                + delta_f_z(m_3y)
                                + (-1) ** j * lateral_delta_f_z_out[i]
                                for j in range(2)
                            ]
                        )
                        for i in range(2)
                    ]
                )
            ],
        )

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
        track_v = track_v_spacial * q_1_dot
        track_a = (
            track_a_spacial * self.track.length**2 * q_1_dot**2
            + track_v_spacial * self.track.length * q_1_ddot
        )

        se3 = self.track.se3_state(q_1 * self.track.length)

        R = se3.rotation
        q = np.hstack([cpin.utils.SE3ToXYZQUAT(se3), q])

        J_e, J_e_dot = self.track.rotation_jacobians(q_1 * self.track.length)

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
                R.T @ (J_e @ track_a[3:6] + J_e_dot @ track_v[3:6]) * q_1_dot,
                q_ddot,
            ]
        )

        torques = cpin.rnea(self.model, self.data, q, v, a, f_ext)

        # TODO check these indicies!
        return (
            torques,
            self.data.v[3],
            (
                self.data.f[3].linear[3],
                self.data.f[3].angular[0],
                self.data.f[3].angular[1],
            ),
        )

    def set_constraints(self, q_1, q_1_dot, q_1_ddot, q_dot, q_ddot, q, f_z, u):

        cpin.forwardKinematics(
            self.model,
            self.data,
            ca.vertcat([cpin.utils.SE3ToXYZQUAT(self.track.se3_state(q_1 * self.track.length)), q]),
            ca.vertcat([q_1_dot, q_dot]),
        )

        v_3 = self.data.v[self.road_lat_id][:3]

        f_ext = [cpin.Force.Zero() for _ in range(self.model.njoints)]

        f_3x, f_3y, m_3z = ca.vertsplit(self.w3e_func(f_z, u, v_3))
        f_xa, f_za, m_ya = ca.vertsplit(self.w6_func(f_z, u, v_3))

        # TODO Check these indicies
        f_ext[3] = cpin.Force(np.array([f_3x, f_3y, 0]), np.array([0, 0, m_3z]))
        f_ext[6] = cpin.Force(np.array([f_xa, 0, f_za]), np.array([0, m_ya, 0]))

        torques, v_3, (f_3z, m_3x, m_3y) = self.rnea(
            q_1, q_1_dot, q_1_ddot, q, q_dot, q_ddot, f_ext
        )
        v_3x, v_3y, v_3z = ca.vertsplit(v_3)

        self.opti.subject_to(self.f_z_func(f_z, u, v_3, f_za, f_3z, m_3x, m_3y, m_ya) == f_z)

        for i in range(3):
            self.opti.subject_to(torques[i] == 0)

        k = self.prop.p(self.prop.s_k) + self.prop.p_karb
        c = self.prop.p(self.prop.s_c)

        self.opti.subject_to(
            torques[3] == -self.prop.p(self.prop.s_k) * q[2] - self.prop.p(self.prop.s_c) * q_dot[2]
        )
        self.opti.subject_to(
            torques[4]
            == -self.prop.p_theta(self.prop.s_k) * q[3]
            - self.prop.p_theta(self.prop.s_c) * q_dot[3]
        )
        self.opti.subject_to(
            torques[5]
            == -self.prop.p_phi(self.prop.s_k) * q[4] - self.prop.p_phi(self.prop.s_c) * q_dot[4]
        )

        # Power limit
        self.opti.subject_to(u[0] * v_3x <= self.prop.e_max)

        # Control bounds
        self.opti.subject_to(self.opti.bounded(0, u[0], 1))
        self.opti.subject_to(self.opti.bounded(0, u[1], 1))
        self.opti.subject_to(self.opti.bounded(-self.prop.g_steer_max, u[2], self.prop.g_steer_max))

        # Track bounds
        n_l, n_r = self.track.state(np.array([q_1 * self.track.length]))[-2:]
        self.opti.subject_to(self.opti.bounded(-n_r, q[0], n_l))

        f_x = ca.vertcat(
            ca.horzcat(*(2 * [u[1] * self.prop.p_kb / 2])),
            ca.horzcat(*(2 * [u[1] * (1 - self.prop.p_kb) + u[0]])) / 2,
        )
        f_y = self.f_y_func(f_z, u, v_3)

        for i in range(2):
            for j in range(2):
                mu_x = (
                    self.prop.t_Dx1
                    + self.prop.t_Dx2 * (f_z[i, j] - self.prop.t_Fznom) / self.prop.t_Fznom
                )
                mu_y = (
                    self.prop.t_Dy1
                    + self.prop.t_Dy2 * (f_z[i, j] - self.prop.t_Fznom) / self.prop.t_Fznom
                )

                self.opti.subject_to(
                    (f_x[i, j] / (mu_x * f_z[i, j])) ** 2 + (f_y[i, j] / (mu_y * f_z[i, j])) ** 2
                    <= 1
                )


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
