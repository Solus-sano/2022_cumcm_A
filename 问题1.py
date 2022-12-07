import numpy as np
from matplotlib import rcParams
from math import *
from sympy.abc import t
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pfun_1(ip,t):
    """第1小问微分方程模型"""
    x,y,z,w=ip
    return np.array([z,
                    w,
                    (-k*(x-y)-beta*(z-w))/m2,
                    (k*(x-y)+beta*(z-w)+f*cos(omega*t)-gama*y-Ita*w)/(m1+m_d)
                    ])

def pfun_2(ip,t):
    """第2小问微分方程模型"""
    x,y,z,w=ip
    return np.array([z,
                    w,
                    (-k*(x-y)-beta*sqrt(abs(z-w))*(z-w))/m2,
                    (k*(x-y)+beta*sqrt(abs(z-w))*(z-w)+f*cos(omega*t)-gama*y-Ita*w)/(m1+m_d)
                    ])

if __name__=='__main__':
    dt=0.01

    m1=4866#浮子质量
    m2=2433#振子质量
    m_d=1335.535#垂荡附加质量 (kg)
    k=80000#弹簧刚度 (N/m)
    beta=10000#平动阻尼系数
    f=6250#垂荡激励力振幅 (N)
    omega=1.4005#入射波浪频率 (s-1)
    T_max=(2*pi/omega)*40#模拟最大时间
    gama=1025*9.8*pi#静水恢复力系数
    Ita=656.3616#垂荡兴波阻尼系数 (N·s/m)

    t_lst=np.arange(0,T_max,dt)

    pfun=pfun_2#选择计算第1小问还是第2小问

    sol=odeint(pfun,[0.0,0.0,0.0,0.0],t_lst)
    rcParams['font.sans-serif']=['SimHei']
    plt.figure()
    plt.plot(t_lst,sol[:,0],label='振子位移')
    plt.plot(t_lst,sol[:,1],label='浮子位移')
    plt.legend()
    plt.figure()
    plt.plot(t_lst,sol[:,2],label='振子速度')
    plt.plot(t_lst,sol[:,3],label='浮子速度')
    plt.legend()


    print('10 s、20 s、40 s、60 s、100 s 时，振子位移：')
    print(sol[int(10/dt),0],'',sol[int(20/dt),0],'',
        sol[int(40/dt),0],'',sol[int(60/dt),0],'',sol[int(100/dt),0])
    print('10 s、20 s、40 s、60 s、100 s 时，浮子位移：')
    print(sol[int(10/dt),1],'',sol[int(20/dt),1],'',
        sol[int(40/dt),1],'',sol[int(60/dt),1],'',sol[int(100/dt),1])
    print('10 s、20 s、40 s、60 s、100 s 时，振子速度：')
    print(sol[int(10/dt),2],'',sol[int(20/dt),2],'',
        sol[int(40/dt),2],'',sol[int(60/dt),2],'',sol[int(100/dt),2])
    print('10 s、20 s、40 s、60 s、100 s 时，浮子速度：')
    print(sol[int(10/dt),3],'',sol[int(20/dt),3],'',
        sol[int(40/dt),3],'',sol[int(60/dt),3],'',sol[int(100/dt),3])
    
    plt.show()