import numpy as np
from matplotlib import rcParams
from math import *
from sympy.abc import t
from scipy.integrate import odeint,quad
import matplotlib.pyplot as plt
from tqdm import tqdm

def power(sol,beta,alpha=1):
    """计算平均功率"""
    t2=250
    E=0;
    x=sol[:,0]
    y=sol[:,1]
    z=sol[:,2]
    w=sol[:,3]
    x_y=np.abs(x-y)
    d_xy=np.diff(x_y)
    d_xy=np.hstack((d_xy,d_xy[-1:]))
    xy_max=np.amax(x_y[int(t2/dt):])
    P=pow(omega,alpha+1)*beta*pow(xy_max,alpha+1)*(2/pi)*quad(lambda x:pow(cos(x),alpha+1),0,pi/2)[0]
    return(P)

def solve_1(beta1,beta2,step):
    ans_lst=[]
    max_beta=0;max_P=0
    def pfun(ip,t):
        """第1小问微分方程模型"""
        x,y,z,w=ip
        return np.array([z,
                        w,
                        (-k*(x-y)-beta*(z-w))/m2,
                        (k*(x-y)+beta*(z-w)+f*cos(omega*t)-gama*y-Ita*w)/(m1+m_d)
                        ])
    t_lst=np.arange(0,T_max,dt)
    print('solving...')
    for beta in tqdm(range(beta1,beta2,step)):
        sol=odeint(pfun,[0.0,0.0,0.0,0.0],t_lst)
        P=power(sol,beta)
        ans_lst.append(P)
        if P>max_P:
            max_beta=beta
            max_P=P
    plt.figure()
    plt.plot(np.arange(beta1,beta2,step),ans_lst)
    plt.xlabel('阻尼系数')
    plt.ylabel('功率')
    plt.legend()
    plt.grid()
    return max_beta,max_P


if __name__=='__main__':
    dt=0.01

    m1=4866#浮子质量
    m2=2433#振子质量
    m_d=1165.992#垂荡附加质量 (kg)
    k=80000#弹簧刚度 (N/m)
    f=4890#垂荡激励力振幅 (N)
    omega=2.2143#入射波浪频率 (s-1)
    T_max=300#模拟最大时间
    gama=1025*9.8*pi#静水恢复力系数
    Ita=167.8395#垂荡兴波阻尼系数 (N·s/m)
    rcParams['font.sans-serif']=['SimHei']

    """多次缩小搜索范围步幅确定最优解"""
    max_beta,p=solve_1(36500,37000,1)
    print("情况1的最优阻尼系数：",max_beta)
    print("情况1的最大功率：",p)
    plt.show()