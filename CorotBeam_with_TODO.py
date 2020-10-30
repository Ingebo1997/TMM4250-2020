# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:43:51 2018

@author: bjohau
"""

import numpy as np
import math


def rot_matrix(theta):
    """
    Return the 2x2 rotation matrix representing a rotation theta
    :param theta:  rotation angle in radians
    :return: Rotation matrix (or tensor)
    """
    s = math.sin(theta)
    c = math.cos(theta)
    R = np.array([[c, -s],
                  [s, c]])
    return R


def beam2local_def_disp(ex, ey, disp_global):
    """

    :param ex: element x coordinate [x1, x2] in undeformed position
    :param ey: element y coordinate [y1, y2] in undeformed position
    :param disp_global:  displacement vector [u1, v1, r1, u2, v2, r2] in global directions
    :return: disp_local_def: displacement vector [u1, v1, r1, u2, v2, r2] in local directions
    """
    element_vector = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    unit_element_vector = element_vector / np.linalg.norm(element_vector)
    L0 = math.sqrt(element_vector @ element_vector)

    # Deformed position and unit vector along element
    ex_def = ex + [disp_global[0], disp_global[3]]
    ey_def = ey + [disp_global[1], disp_global[4]]

    # The deformed beam 
    element_vector_def_x = np.array([ex_def[1] - ex_def[0], ey_def[1] - ey_def[0]])
    # Length of deformed beam
    unit_element_vector_def_x = element_vector_def_x / np.linalg.norm(element_vector_def_x)
    # print("unitvector of x", unit_element_vector_def_x)
    Ld = math.sqrt(element_vector_def_x @ element_vector_def_x)

    # Ortogonal of deformed beam
    unit_element_vector_def_y = np.array([-unit_element_vector_def_x[1], unit_element_vector_def_x[0]])

    R1 = rot_matrix(disp_global[2])
    R2 = rot_matrix(disp_global[5])

    t_1 = R1 @ unit_element_vector
    t_2 = R2 @ unit_element_vector

    theta1_def = math.asin(unit_element_vector_def_y.T @ t_1)
    theta2_def = math.asin(unit_element_vector_def_y.T @ t_2)

    def_disp_local = np.array([-0.5 * (Ld - L0),
                               0.0,
                               theta1_def,
                               0.5 * (Ld - L0),
                               0.0,
                               theta2_def])
    return def_disp_local


def beam2corot_Ke_and_Fe(ex, ey, ep, disp_global):
    """
    Compute the stiffness matrix and internal forces for a two dimensional beam element
    relative to deformed configuration.
    
    :param list ex: element x coordinates [x1, x2]
    :param list ey: element y coordinates [y1, y2]
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia
    :param list disp_global: displacement vector for the element [tx1,ty1,rz1,tx2,ty2,rz2]


    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: element stiffness matrix [6 x 1]
    """
    # Undeformed length and unit vector along element
    eVec12 = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    L0 = math.sqrt(eVec12 @ eVec12)
    eVec12 /= L0

    # Deformed position and unit vector along element
    ex_def = ex + [disp_global[0], disp_global[3]]
    ey_def = ey + [disp_global[1], disp_global[4]]

    Ke_local = beam2local_stiff(L0, ep)
    Te_local = beam2corot_Te(ex_def, ey_def)
    v_local = beam2local_def_disp(ex, ey, disp_global)

    fe_int_local = Ke_local @ v_local
    fe_int_global = Te_local.T @ fe_int_local

    fe = fe_int_local / L0

    Kg_local = np.array([
        [0, fe[1] / 2., 0., 0., -fe[1] / 2., 0.],
        [fe[1] / 2., -fe[0], 0., fe[1] / 2., fe[0], 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., -fe[1] / 2., 0., 0., fe[1] / 2., 0.],
        [-fe[1] / 2., fe[0], 0., fe[1] / 2., -fe[0], 0.],
        [0., 0., 0., 0., 0., 0.]
    ])

    # Ke_global = Te_local.T @ (Ke_local + Kg_local) @ Te_local
    Ke_global = Te_local.T @ (Ke_local) @ Te_local
    # print(disp_global)

    return Ke_global, fe_int_global


def beam2corot_Te(ex, ey):
    """
    Compute the transformation matrix for an element
    
    :param list ex: element x coordinates [x1, x2]
    :param list ey: element y coordinates [y1, y2]
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia   
    :param list eq: distributed loads, local directions [qx, qy]
    :return mat Te: element transformation from global to local
    """
    # print("The x vector", ex)
    # print("The y vector", ey)
    n = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    # print("n is ", n)
    L = np.linalg.norm(n)
    n = n / L
    if math.isnan(n[0]):
        print(L)
        raise Exception("There is a Nan among our midsts")
        


    Te = np.array([
        [n[0], n[1], 0., 0., 0., 0.],
        [-n[1], n[0], 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., n[0], n[1], 0.],
        [0., 0., 0., -n[1], n[0], 0.],
        [0., 0., 0., 0., 0., 1.]
    ])

    return Te


def beam2local_stiff(L, ep):
    """
    Compute the stiffness matrix for a two dimensional beam element.
    
    :param list L : element length
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia   
    :return mat Kle: element stiffness matrix [6 x 6]
    """

    E = ep[0]
    A = ep[1]
    I = ep[2]

    Kle = np.array([
        [E * A / L, 0., 0., -E * A / L, 0., 0.],
        [0., 12 * E * I / L ** 3., 6 * E * I / L ** 2., 0., -12 * E * I / L ** 3., 6 * E * I / L ** 2.],
        [0., 6 * E * I / L ** 2., 4 * E * I / L, 0., -6 * E * I / L ** 2., 2 * E * I / L],
        [-E * A / L, 0., 0., E * A / L, 0., 0.],
        [0., -12 * E * I / L ** 3., -6 * E * I / L ** 2., 0., 12 * E * I / L ** 3., -6 * E * I / L ** 2.],
        [0., 6 * E * I / L ** 2., 2 * E * I / L, 0., -6 * E * I / L ** 2., 4 * E * I / L]
    ])

    return Kle


def beam2e(ex, ey, ep, eq=None):
    """
    Compute the linear stiffness matrix for a two dimensional beam element.
    Largely from CALFEM core module

    :param list ex: element x coordinates [x1, x2]
    :param list ey: element y coordinates [y1, y2]
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia
    :param list eq: distributed loads, local directions [qx, qy]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: element stiffness matrix [6 x 1] (if eq!=None)
    """

    n = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    L = np.linalg.norm(n)
    n = n / L

    qx = 0.
    qy = 0.
    if not eq is None:
        qx = eq[0]
        qy = eq[1]

    Kle = beam2local_stiff(L, ep)

    fle = L * np.mat([qx / 2, qy / 2, qy * L / 12, qx / 2, qy / 2, -qy * L / 12]).T

    Te = beam2corot_Te(ex, ey)

    Ke = Te.T @ Kle @ Te
    fe = Te.T @ fle

    if eq is None:
        return Ke
    else:
        return Ke, fe

# print(beam2local_def_disp([0,3],[0,3],[4,3,7,6,-3,0]))
