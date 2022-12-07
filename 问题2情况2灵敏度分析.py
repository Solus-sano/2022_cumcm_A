import numpy as np
from matplotlib import rcParams
from math import *
from sympy.abc import t
from scipy.integrate import odeint,quad
import matplotlib.pyplot as plt
from tqdm import tqdm

def power(sol,beta,alpha=1):
    """计算平均功率"""
    t2=250# t2时间后震动趋于稳定
    x=sol[:,0]
    y=sol[:,1]
    x_y=np.abs(x-y)
    d_xy=np.diff(x_y)
    d_xy=np.hstack((d_xy,d_xy[-1:]))
    xy_max=np.amax(x_y[int(t2/dt):])
    P=pow(omega,alpha+2)*beta*pow(xy_max,alpha+2)*(2/pi)*quad(lambda x:pow(cos(x),alpha+2),0,pi/2)[0]
    return(P)

def sens_beta(best_beta,best_alpha,beta_e=10):
    print("阻尼系数：")
    p_lst=[]
    alpha=best_alpha;beta=0
    def pfun(ip,t):
        """第2小问微分方程模型"""
        x,y,z,w=ip
        def sgn(a):
            if a>0:return 1
            elif a<0: return -1
            else: return 0
        return np.array([z,
                        w,
                        (-k*(x-y)-beta*pow(abs(z-w),alpha+1)*sgn(z-w))/m2,
                        (k*(x-y)+beta*pow(abs(z-w),alpha+1)*sgn(z-w)+f*cos(omega*t)-gama*y-Ita*w)/(m1+m_d)
                        ])
    t_lst=np.arange(0,T_max,dt)
    for i in tqdm(np.linspace(best_beta-beta_e,best_beta+beta_e,100)):
        beta=i
        sol=odeint(pfun,[0.0,0.0,0.0,0.0],t_lst)
        P=power(sol,beta,alpha)
        p_lst.append(P)
    alpha,beta=best_alpha,best_beta
    sol=odeint(pfun,[0.0,0.0,0.0,0.0],t_lst)
    P=power(sol,beta,alpha)
    plt.figure()
    plt.scatter(best_beta,P,label='最优参数处功率')
    plt.plot(np.linspace(best_beta-beta_e,best_beta+beta_e,100),p_lst)
    plt.xlabel('beta')
    plt.ylabel('功率')
    plt.legend()
    plt.grid()

def sens_alpha(best_beta,best_alpha,alpha_e=0.001):
    print("指数系数：")
    p_lst=[]
    alpha=0;beta=best_beta
    def pfun(ip,t):
        """第2小问微分方程模型"""
        x,y,z,w=ip
        def sgn(a):
            if a>0:return 1
            elif a<0: return -1
            else: return 0
        return np.array([z,
                        w,
                        (-k*(x-y)-beta*pow(abs(z-w),alpha+1)*sgn(z-w))/m2,
                        (k*(x-y)+beta*pow(abs(z-w),alpha+1)*sgn(z-w)+f*cos(omega*t)-gama*y-Ita*w)/(m1+m_d)
                        ])
    t_lst=np.arange(0,T_max,dt)
    for i in tqdm(np.linspace(best_alpha-alpha_e,best_alpha+alpha_e,100)):
        alpha=i
        sol=odeint(pfun,[0.0,0.0,0.0,0.0],t_lst)
        P=power(sol,beta,alpha)
        p_lst.append(P)
    alpha,beta=best_alpha,best_beta
    sol=odeint(pfun,[0.0,0.0,0.0,0.0],t_lst)
    P=power(sol,beta,alpha)
    plt.figure()
    plt.scatter(best_alpha,P,label='最优参数处功率')
    plt.plot(np.linspace(best_alpha-alpha_e,best_alpha+alpha_e,100),p_lst)
    plt.xlabel('alpha')
    plt.ylabel('功率')
    plt.legend()
    plt.grid()

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

    sens_beta(98254.8371371,0.458481)
    sens_alpha(98254.8371371,0.458481) 
    plt.show()