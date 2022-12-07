import numpy as np
from matplotlib import rcParams
from math import *
from sympy.abc import t
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pfun(ip,t):
    """第3问微分方程模型"""
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
                [k2*(th2-th1)+beta_2*(ph2-ph1)-Ita_2*ph1-(((2+15*(2-y)**2)/(8+30*(2-y)))*((m_v+m_f)*g-1025*g*pi*y)+gama_2)*th1+L*cos(omega*t)],
                [m_v*g*(l0+x+0.5*h_v)*sin(th2)-2*z*ph2*m_v*(l0+x+0.5*h_v)-k2*(th2-th1)-beta_2*(ph2-ph1)]])
    Op=np.dot(np.linalg.inv(A),b)
    return Op.reshape((-1,))



if __name__=='__main__':
    dt=0.01
    g=9.8

    l0=0.2019575#弹簧初始长度
    m_f=4866#浮子质量
    m_v=2433#振子质量
    m_d=1028.876#垂荡附加质量 (kg)
    J_d=7001.914#纵摇附加转动惯量 (kg·m^2)
    J_v=202.75#振子绕质心转动惯量
    J_f=16137.73119#浮子转动惯量
    k1=80000#弹簧刚度 (N/m)
    k2=250000#扭转弹簧刚度 (N·m)
    beta_1=10000#直线阻尼器阻尼系数
    beta_2=1000#旋转阻尼器阻尼系数
    f=3640#垂荡激励力振幅 (N)
    L=1690#纵摇激励力矩振幅 (N·m)
    omega=1.7152#入射波浪频率 (s-1)
    T_max=(2*pi/omega)*40#模拟最大时间
    gama_1=1025*g*pi#静水恢复力系数
    gama_2=8890.7#静水恢复力矩系数 (N·m)
    Ita_1=683.4558#垂荡兴波阻尼系数 (N·s/m)
    Ita_2=654.3383#纵摇兴波阻尼系数 (N·m·s)
    h_v=0.5#振子高度

    t_lst=np.arange(0,T_max,dt)


    sol=odeint(pfun,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],t_lst)

    """将振子相对铰链中心的径向位移、速度转换成垂荡位移、垂荡速度"""
    xx=np.zeros(sol.shape[0])
    vx=np.zeros(sol.shape[0])
    for idx in range(xx.shape[0]):
        xx[idx]=(l0+sol[idx,0])*cos(sol[idx,5])-l0+sol[idx,1]
        vx[idx]=sol[idx,2]*cos(sol[idx,5])+sol[idx,3]


    rcParams['font.sans-serif']=['SimHei']
    plt.figure()
    plt.title('')
    plt.plot(t_lst,xx,label='振子垂荡位移')
    plt.plot(t_lst,sol[:,1],label='浮子垂荡位移')
    plt.legend()

    plt.figure()
    plt.plot(t_lst,vx,label='振子垂荡速度')
    plt.plot(t_lst,sol[:,3],label='浮子垂荡速度')
    plt.legend()

    plt.figure()
    plt.plot(t_lst,sol[:,4],label='浮子纵摇角度')
    plt.plot(t_lst,sol[:,5],label='振子纵摇角度')
    plt.legend()

    plt.figure()
    plt.plot(t_lst,sol[:,6],label='浮子纵摇角速度')
    plt.plot(t_lst,sol[:,7],label='振子纵摇角速度')
    plt.legend()

    print('10 s、20 s、40 s、60 s、100 s 时，振子垂荡位移：')
    print(xx[int(10/dt)],'',xx[int(20/dt)],'',xx[int(40/dt)],
    '',xx[int(60/dt)],'',xx[int(100/dt)])
    print('10 s、20 s、40 s、60 s、100 s 时，浮子垂荡位移：')
    print(sol[int(10/dt),1],'',sol[int(20/dt),1],'',sol[int(40/dt),1],
    '',sol[int(60/dt),1],'',sol[int(100/dt),1])
    print('10 s、20 s、40 s、60 s、100 s 时，振子垂荡速度：')
    print(vx[int(10/dt)],'',vx[int(20/dt)],'',vx[int(40/dt)],
    '',vx[int(60/dt)],'',vx[int(100/dt)])
    print('10 s、20 s、40 s、60 s、100 s 时，浮子垂荡速度：')
    print(sol[int(10/dt),3],'',sol[int(20/dt),3],'',sol[int(40/dt),3],
    '',sol[int(60/dt),3],'',sol[int(100/dt),3])
    print('10 s、20 s、40 s、60 s、100 s 时，浮子纵摇角度：')
    print(sol[int(10/dt),4],'',sol[int(20/dt),4],'',sol[int(40/dt),4],
    '',sol[int(60/dt),4],'',sol[int(100/dt),4])
    print('10 s、20 s、40 s、60 s、100 s 时，振子纵摇角度：')
    print(sol[int(10/dt),5],'',sol[int(20/dt),5],'',sol[int(40/dt),5],
    '',sol[int(60/dt),5],'',sol[int(100/dt),5])
    print('10 s、20 s、40 s、60 s、100 s 时，浮子纵摇角速度：')
    print(sol[int(10/dt),6],'',sol[int(20/dt),6],'',sol[int(40/dt),6],
    '',sol[int(60/dt),6],'',sol[int(100/dt),6])
    print('10 s、20 s、40 s、60 s、100 s 时，振子纵摇角速度：')
    print(sol[int(10/dt),7],'',sol[int(20/dt),7],'',sol[int(40/dt),7],
    '',sol[int(60/dt),7],'',sol[int(100/dt),7])
    
    plt.show()