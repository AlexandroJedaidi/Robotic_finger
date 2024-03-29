# L -26, here we will calculate the torque of the joints
import numpy as np

def system_torque(
                  th1, th2, th3, a1, a2, a3, th1dot, th2dot,th3dot,
                  m1, m2, m3, x1, x2, z1, x3, y3, z3,
                  y1,y2, z2, th1ddot, th2ddot,th3ddot, ixx1, iyy1,
                  izz1, ixy1, ixz1, iyz1, ixx2, iyy2, izz2, ixy2, ixz2, iyz2,
                  ixx3, ixy3, iyy3, iyz3, ixz3, izz3, g=9.8):


    '''
    Joints - Two types - Prismatic, Revolute we have only revolute joints here.
    system torque will provide the torque for both the joints tau1 and tau2.
    dh_table is the DH para,meters of our finger model.
    th1- Angle of first link from ground (between x0, x1)
    th2- Angle of 2nd link with respect to first link ( between x1,x2)
    a1-  First link length
    a2 - second link length
    th1dot - first joint angular velocity
    th2dot - second joint angular velocity
    m1 - mass of the first link
    m2 - mass of teh second link
    th1ddot - angular acceleration of the first joint
    th2ddot - angular acceleration of second joint
    ixx1, iyy1, izz1, ixy1, ixz1, iyz1 - inertia tensors of link 1
    ixx2, iyy2, izz2, ixy2, ixz2, iyz2 - inertia tensors of link 2
    dh_table = [ alpha, a, theta, d]
    alpha - link twist angle between zi-1 to zi along xi-1
    a - link lengths
    theta -  (joint angles) angle between xi to xi-1 along zi
    d - distances between xi and xi-1 along zi
    t - Total transformation matrix
    tdh - Transformation matrix of each joint frame
    x1, y1, z1 - location of center of mass of link 1
    x2, y2, z2 - location of center of mass of link 2
    we calculate DH parameters to get the relative information
    between 2 joints
    '''

    # above we have defined the DH table to get the dh parameters for the system

    dh_table = np.array([
        [0, 0, th1, 0],
        [0, a1, th2, 0],
        [0, a3, th3, 0],
        [0, a2, 0, 0]
            ])

    # Extracting all the respective values of alpha, a, theta, d from dh_table matrix
    alpha = dh_table[:, 0]
    a = dh_table[:, 1]
    theta = dh_table[:, 2]
    d = dh_table[:, 3]
    t = np.eye(4)

    # alpha = np.empty((3,1))
    # a = np.empty((3,1))
    # theta = np.empty((3,1))
    # d = np.empty((3,1))
    # A = np.empty()

    n = 3  # number of joint frames including end effector
    tdh = np.empty((4, 4, 4))  # defining tdh "Arm matrix" 4 arm matrices
    for i in range(n):
        tdh[i] = np.array([
                   [np.cos(theta[i]), -np.sin(theta[i]), 0, a[i]],
                   [np.cos(alpha[i])*np.sin(theta[i]), np.cos(alpha[i])*np.cos(theta[i]), -np.sin(alpha[i]), -d[i]*np.sin(alpha[i])],
                   [np.sin(alpha[i])*np.sin(theta[i]), np.sin(alpha[i])*np.cos(theta[i]), np.cos(alpha[i]), d[i]*np.cos(alpha[i])],
                   [0, 0, 0, 1]
                   ])
        t = t @ tdh[i]
    t0_1 = tdh[0]  # TRANSFORMATION MATRIX -1
    t1_2 = tdh[1]  # TRANSFORMATION MATRIX -2
    t2_3 = tdh[2]  # TRANSFORMATION MATRIX -3
    t3_4 = tdh[3]  # TRANSFORMATION MATRIX -4
    # # position vectors
    # a = t[:, 3]
    # p = a[:, 2]

    # Velocity kinematics
    r01 = t0_1[0:3, 0:3]  # rotation matrix of joint 1
    p01 = t0_1[0:3, 3]    # vector from the origin of frame 0 to the origin of frame 1 position vector
    r12 = t1_2[0:3, 0:3]  # rotation matrix of joint 2
    p12 = t1_2[0:3, 3]    # vector from the origin of frame 1 to the origin of frame 2
    r23 = t2_3[0:3, 0:3]  # rotation matrix of joint 3
    p23 = t2_3[0:3, 3]    # vector from the origin of frame 0 to the origin of frame 1
    r34 = t3_4[0:3, 0:3]
    p34 = t3_4[0:3, 3]

    # Angular velocity propagation

    w0 = np.array([0, 0, 0])
    w1 = (r01 @ w0 + np.array([0, 0, th1dot]))
    w2 = (r12 @ w1 + np.array([0, 0, th2dot]))
    w3 = (r23 @ w2 + np.array([0,0, th3dot]))
    w4 = (r34 @ w3)

    # End effector angular velocity w.r.t base
    w04 = r01 @ r12 @ r23 @ r34

    # Linear velocity propagation

    v0 = np.array([0, 0, 0])
    v1 = r01 @ (v0 + np.cross(w0, p01))
    v2 = r12 @ (v1 + np.cross(w1, p12))
    v3 = r23 @ (v2 + np.cross(w2, p23))
    v4 = r34 @ (v3 + np.cross(w3,p34))
    # End effector linear velocity

    v04 = r01 @ r12 @ r23 @ r34 @ v4

    # Dynamic model

    # Location of center of mass of all 3 the links

    pc1 = np.array([x1, y1, z1])
    pc2 = np.array([x2, y2, z2])
    pc3 = np.array([x3, y3, z3])


    # Angular acceleration vectors

    al0 = np.array([0, 0, 0])
    al1 = r01 @ (al0+np.cross(w0, np.array([0, 0, th1dot]))) + np.array([0, 0, th1ddot])
    al2 = r12 @ (al1+np.cross(w1, np.array([0, 0, th2dot]))) + np.array([0, 0, th2ddot])
    al3 = r23 @ (al2+np.cross(w2, np.array([0, 0, th3dot]))) + np.array([0, 0, th3ddot])
    al4 = r34 @ al3

    # Linear accelerations

    a0 = np.array([0, g, 0])  # Here we have considered that in the base only gravitational acceleration is acting
    a1 = r01 @ (a0 + np.cross(al0, p01)) + np.cross(w0, np.cross(w0, p01))
    a2 = r12 @ (a1 + np.cross(al1, p12)) + np.cross(w1, np.cross(w1, p12))
    a3 = r23 @ (a2 + np.cross(al2, p23)) + np.cross(w2, np.cross(w2, p23))
    a4 = r34 @ (a3 + np.cross(al3,p34)) + np.cross(w3,np.cross(w3, p34))

    # linear acceleration of centre of mass of links

    ac1 = a1 + np.cross(al1, pc1) + np.cross(w1, np.cross(w1, pc1))
    ac2 = a2 + np.cross(al2, pc2) + np.cross(w2, np.cross(w2, pc2))
    ac3 = a3 + np.cross(al3, pc3) + np.cross(w3, np.cross(w3, pc3))

    # Inertial forces of the links

    if1 = m1 * ac1
    if2 = m2 * ac2
    if3 = m3 * ac3

    # Defining the inertia matrix for both the links

    i1 = np.array([[ixx1, ixy1, ixz1], [ixy1, iyy1, iyz1], [ixz1, iyz1, izz1]])
    i2 = np.array([[ixx2, ixy2, ixz2], [ixy2, iyy2, iyz2], [ixz2, iyz2, izz2]])
    i3 = np.array([[ixx3, ixy3, ixz3], [ixy3, iyy3, iyz3], [ixz3, iyz3, izz3]])

    # torques acting on the links

    tau_1 = (i1 @ al1) + np.cross(w1, (i1 @ w1))
    tau_2 = (i2 @ al2) + np.cross(w2, (i2 @ w2))
    tau_3 = (i3 @ al3) + np.cross(w3, (i3 @ w3))

    # Calculating the Joint forces and joint torques from link 2 to link 0

    f4 = np.array([0, 0, 0])
    n4 = np.array([0, 0, 0])

    # Joint forces
    f3 = r34 @ f4 + if3
    f2 = r23 @ f3 + if2
    f1 = r12 @ f2 + if1
    f0 = r01 @ f1

    # Joint Moments

    n3 = (r34 @ n4) + np.cross(pc3, if3) + np.cross(p34, (r34 @ f4)) + tau_3
    n2 = (r23 @ n3) + np.cross(pc2, if2) + np.cross(p23, (r23 @ f3)) + tau_2
    n1 = (r12 @ n2) + np.cross(pc1, if1) + np.cross(p12, (r12 @ f2)) + tau_1
    n0 = (r01 @ n1) + np.cross(p01, (r01 @ f1))

    # Final joint torques acting along z axis here we will take the zth component of n1 and n2

    j_tau_1 = n1[2]
    j_tau_2 = n2[2]
    j_tau_3 = n3[2]

    return j_tau_1, j_tau_2, j_tau_3

