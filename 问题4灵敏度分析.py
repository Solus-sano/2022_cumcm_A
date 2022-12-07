import numpy as np
from matplotlib import rcParams
from math import *
from sympy.abc import t
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm

def power(sol,beta_1,beta_2):
    """计算平均功率"""
    t1=300;t2=350# t1时间后震动趋于稳定
    vx_r=sol[:,2]
    oth_r=sol[:,6]-sol[:,7]
    idx1=int(t1/dt);idx2=int(t2/dt)
    E1,E2=0,0
    for i in range(idx1,idx2):
        E1+=(beta_1*vx_r[i]**2)*dt
        E2+=(beta_2*oth_r[i]**2)*dt
    P1=E1/(t2-t1)
    P2=E2/(t2-t1)
    # print('垂荡阻尼功率',P1,' 旋转阻尼功率：',P2)
    P=P1+P2
    return P


def sens_beta_1(best_beta_1,best_beta_2,beta_1_e=100):
    print("平动阻尼系数：")
    p_lst=[]
    max_P=0
    beta_1=0;beta_2=best_beta_2
    def pfun(ip,t):
        """第1小问微分方程模型"""
        x,y,z,w,th1,th2,ph1,ph2=ip
        A=np.array([[1,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0],
                    [0,0,m_v,m_v*cos(th2),0,0,0,0],
                    [0,0,0,m_f+m_d,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,J_f+J_d,0],
                    [0,0,0,-m_v*(l0+x+0.5*h_v)*sin(th2),0,0,0,J_v+m_v*(l0+x+0.5*h_v)**2]])
        b=np.array([[z],
                    [w],
                    [m_v*g*(1-cos(th2))+m_v*(l0+x+0.5*h_v)*ph2**2-k1*x-beta_1*z],
                    [m_v*g*(1-cos(th2))+(k1*x+beta_1*z)*cos(th2)+f*cos(omega*t)-Ita_1*w-gama_1*y],
                    [ph1],
                    [ph2],
                    [k2*(th2-th1)+beta_2*(ph2-ph1)-Ita_2*ph1-(((2+15*(2-y)**2)/(8+30*(2-y)))*((m_v+m_f)*g-1025*pi*y)+gama_2)*th1+L*cos(omega*t)],
                    [m_v*g*(l0+x+0.5*h_v)*sin(th2)-2*z*ph2*m_v-k2*(th2-th1)-beta_2*(ph2-ph1)]])
        Op=np.dot(np.linalg.inv(A),b)
        return Op.reshape((-1,))

    t_lst=np.arange(0,T_max,dt)
    for i in tqdm(np.linspace(best_beta_1-beta_1_e,best_beta_1+beta_1_e,20)):
        beta_1=i
        sol=odeint(pfun,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],t_lst)
        P=power(sol,beta_1,beta_2)
        max_P=max(P,max_P)
        p_lst.append(P)

    beta_1,beta_2=best_beta_1,best_beta_2
    sol=odeint(pfun,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],t_lst)
    P=power(sol,beta_1,beta_2)
    plt.figure()
    plt.scatter(best_beta_1,P,label='最优参数处功率')
    plt.plot(np.linspace(best_beta_1-beta_1_e,best_beta_1+beta_1_e,20),p_lst)
    plt.xlabel('beta_1')
    plt.ylabel('功率')
    plt.ylim((317.8,317.9))
    plt.legend()
    plt.grid()

def sens_beta_2(best_beta_1,best_beta_2,beta_2_e=100):
    print("旋转阻尼系数：")
    p_lst=[]
    max_P=0
    beta_1=best_beta_1;beta_2=0
    def pfun(ip,t):
        """第1小问微分方程模型"""
        x,y,z,w,th1,th2,ph1,ph2=ip
        A=np.array([[1,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0],
                    [0,0,m_v,m_v*cos(th2),0,0,0,0],
                    [0,0,0,m_f+m_d,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,J_f+J_d,0],
                    [0,0,0,-m_v*(l0+x+0.5*h_v)*sin(th2),0,0,0,J_v+m_v*(l0+x+0.5*h_v)**2]])
        b=np.array([[z],
                    [w],
                    [m_v*g*(1-cos(th2))+m_v*(l0+x+0.5*h_v)*ph2**2-k1*x-beta_1*z],
                    [m_v*g*(1-cos(th2))+(k1*x+beta_1*z)*cos(th2)+f*cos(omega*t)-Ita_1*w-gama_1*y],
                    [ph1],
                    [ph2],
                    [k2*(th2-th1)+beta_2*(ph2-ph1)-Ita_2*ph1-(((2+15*(2-y)**2)/(8+30*(2-y)))*((m_v+m_f)*g-1025*pi*y)+gama_2)*th1+L*cos(omega*t)],
                    [m_v*g*(l0+x+0.5*h_v)*sin(th2)-2*z*ph2*m_v-k2*(th2-th1)-beta_2*(ph2-ph1)]])
        Op=np.dot(np.linalg.inv(A),b)
        return Op.reshape((-1,))

    t_lst=np.arange(0,T_max,dt)
    for i in tqdm(np.linspace(best_beta_2-beta_2_e,best_beta_2+beta_2_e,20)):
        beta_2=i
        sol=odeint(pfun,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],t_lst)
        P=power(sol,beta_1,beta_2)
        p_lst.append(P)

    beta_1,beta_2=best_beta_1,best_beta_2
    sol=odeint(pfun,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],t_lst)
    P=power(sol,beta_1,beta_2)
    plt.figure()
    plt.scatter(best_beta_2,P,label='最优参数处功率')
    plt.plot(np.linspace(best_beta_2-beta_2_e,best_beta_2+beta_2_e,20),p_lst)
    plt.xlabel('beta_2')
    plt.ylabel('功率')
    plt.ylim((317.8,317.9))
    plt.legend()
    plt.grid()

if __name__=='__main__':
    rcParams['font.sans-serif']=['SimHei']
    dt=0.01
    g=9.8

    l0=0.2019575#弹簧初始长度
    m_f=4866#浮子质量
    m_v=2433#振子质量
    m_d=1091.099#垂荡附加质量 (kg)
    J_d=7142.493#纵摇附加转动惯量 (kg·m^2)
    J_v=202.75#振子绕质心转动惯量
    J_f=16137.73119#浮子转动惯量
    k1=80000#弹簧刚度 (N/m)
    k2=250000#扭转弹簧刚度 (N·m)
    f=1760#垂荡激励力振幅 (N)
    L=2140#纵摇激励力矩振幅 (N·m)
    omega=1.9806#入射波浪频率 (s-1)
    T_max=400#(2*pi/omega)*40#模拟最大时间
    gama_1=1025*g*pi#静水恢复力系数
    gama_2=8890.7#静水恢复力矩系数 (N·m)
    Ita_1=528.5018#垂荡兴波阻尼系数 (N·s/m)
    Ita_2=1655.909#纵摇兴波阻尼系数 (N·m·s)
    h_v=0.5#振子高度

    # sens_beta_1(58636.28387451172,93404.86526489258)
    sens_beta_2(58636.28387451172,93404.86526489258)

    plt.show()