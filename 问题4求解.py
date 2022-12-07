import numpy as np
from matplotlib import rcParams
from math import *
from sympy.abc import t
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import geatpy as ea


def abcpower(sol,beta_1,beta_2):
    """计算平均功率"""
    t1=300;t2=350# t1时间后震动趋于稳定
    vx_r=sol[:,2]
    oth_r=sol[:,6]-sol[:,7]
    idx1=int(t1/dt);idx2=int(t2/dt)
    # print('位移振幅',0.5*(np.amax(sol[idx2:,0])-np.amin(sol[idx2:,0])))
    # print('角度振幅',0.5*(np.amax(sol[idx2:,4]-sol[idx2:,5])-np.amin(sol[idx2:,4]-sol[idx2:,5])))
    E1,E2=0,0
    for i in range(idx1,idx2):
        E1+=(beta_1*vx_r[i]**2)*dt
        E2+=(beta_2*oth_r[i]**2)*dt
    P1=E1/(t2-t1)
    P2=E2/(t2-t1)
    # print('垂荡阻尼功率',P1,' 旋转阻尼功率：',P2)
    P=P1+P2
    return P


@ea.Problem.single
def simulate(a):
    beta_1 = a[0]
    beta_2 = a[1]
    t_lst=np.arange(0,T_max,dt)
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

    sol=odeint(pfun,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],t_lst)
    idx1=int(3/dt);idx2=int(350/dt)
    # rcParams['font.sans-serif']=['SimHei']
    P=abcpower(sol,beta_1,beta_2)
    # print("总功率：",P,'\n')
    return P, [-1, -1]

if __name__=='__main__':
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
    # beta_1=10000#直线阻尼器阻尼系数
    # beta_2=1000#旋转阻尼器阻尼系数
    f=1760#垂荡激励力振幅 (N)
    L=2140#纵摇激励力矩振幅 (N·m)
    omega=1.9806#入射波浪频率 (s-1)
    T_max=400#(2*pi/omega)*40#模拟最大时间
    gama_1=1025*g*pi#静水恢复力系数
    gama_2=8890.7#静水恢复力矩系数 (N·m)
    Ita_1=528.5018#垂荡兴波阻尼系数 (N·s/m)
    Ita_2=1655.909#纵摇兴波阻尼系数 (N·m·s)
    h_v=0.5#振子高度

    # simulate(34000,88000)
    t_start=time()

    max_beta_1=0;max_beta_2=0;max_P=-1
    beta_1_lst=np.arange(20000,40000,2000)
    beta_2_lst=np.arange(70000,100000,3000)
    z=np.zeros((beta_1_lst.shape[0],beta_2_lst.shape[0]))
    
    # print(simulate([50000, 50000]))
    
    
    problem = ea.Problem(name='soea quick start demo',
                         M=1,  # 目标维数
                         maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                         Dim=2,  # 决策变量维数
                         varTypes=[0, 0],  # 决策变量的类型列表，0：实数；1：整数
                         lb=[0, 0],  # 决策变量下界
                         ub=[100000, 100000],  # 决策变量上界
                         evalVars=simulate)
    
    algorithm = ea.soea_SEGA_templet(problem,
                                     ea.Population(Encoding='RI', NIND=20),
                                     MAXGEN=50,  # 最大进化代数。
                                     logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                     trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                     maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    
    res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName=r'geat_result')
    
    
    print(res)