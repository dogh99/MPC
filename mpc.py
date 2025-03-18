from celluloid import Camera # 保存动图时用，pip install celluloid
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy
import math


# mpc parameters
NX = 3  # x = x, y, yaw 状态变量的维度
NU = 2  # u = [v,delta] 控制变量的维度
T = 8  # horizon length 预测时域长度
R = np.diag([0.1, 0.1])  # input cost matrix
Rd = np.diag([0.1, 0.1])  # input difference cost matrix
Q = np.diag([1, 1, 1])  # state cost matrix
Qf = Q  # state final matrix


#车辆
dt = 0.1  # 时间间隔，单位：s
L = 2  # 车辆轴距，单位：m
v = 2  # 初始速度
x_0 = 0  # 初始x
y_0 = -3  # 初始y
psi_0 = 0  # 初始航向角

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]最大转向角度
MAX_DSTEER = np.deg2rad(45.0)  # maximum steering speed [rad/s]最大转向速度

MAX_VEL = 2.0  # maximum accel [m/s]

def get_nparray_from_matrix(x):#将矩阵转换为数组
    return np.array(x).flatten()


class KinematicModel_3:
  """假设控制量为转向角delta_f和加速度a
  """

  def __init__(self, x, y, psi, v, L, dt):#初始化函数
    self.x = x
    self.y = y
    self.psi = psi
    self.v = v
    self.L = L
    # 实现是离散的模型
    self.dt = dt#时间间隔
  def update_state(self, a, delta_f):#状态更新函数,其中两个控制量为加速度a和转向角delta_f
    self.x = self.x+self.v*math.cos(self.psi)*self.dt#x方向的位移
    self.y = self.y+self.v*math.sin(self.psi)*self.dt#y方向的位移
    self.psi = self.psi+self.v/self.L*math.tan(delta_f)*self.dt#????航向角迭代公式
    self.v = self.v+a*self.dt

  def get_state(self):#返回车辆的状态
    return self.x, self.y, self.psi, self.v

  def state_space(self, ref_delta, ref_yaw):
    """将模型离散化后的状态空间表达

    Args:
        ref_delta (_type_): 参考的转角控制量
        ref_yaw (_type_): 参考的偏航角

    Returns:
        _type_: _description_
    """

    A = np.matrix([
        [1.0, 0.0, -self.v*self.dt*math.sin(ref_yaw)],
        [0.0, 1.0, self.v*self.dt*math.cos(ref_yaw)],
        [0.0, 0.0, 1.0]])

    B = np.matrix([
        [self.dt*math.cos(ref_yaw), 0],
        [self.dt*math.sin(ref_yaw), 0],
        [self.dt*math.tan(ref_delta)/self.L, self.v*self.dt /(self.L*math.cos(ref_delta)*math.cos(ref_delta))]
    ])

    C = np.eye(3)
    return A, B, C


class MyReferencePath:
    def __init__(self):
        # set reference trajectory
        # refer_path包括4维：位置x, 位置y， 轨迹点的切线方向, 曲率k
        self.refer_path = np.zeros((1000, 4))
        self.refer_path[:, 0] = np.linspace(0, 100, 1000)  # x
        self.refer_path[:, 1] = 2*np.sin(self.refer_path[:, 0]/3.0) + \
            2.5*np.cos(self.refer_path[:, 0]/2.0)  # y
        # 使用差分的方式计算路径点的一阶导和二阶导，从而得到切线方向和曲率
        for i in range(len(self.refer_path)):
            if i == 0:
                dx = self.refer_path[i+1, 0] - self.refer_path[i, 0]
                dy = self.refer_path[i+1, 1] - self.refer_path[i, 1]
                ddx = self.refer_path[2, 0] + \
                    self.refer_path[0, 0] - 2*self.refer_path[1, 0]
                ddy = self.refer_path[2, 1] + \
                    self.refer_path[0, 1] - 2*self.refer_path[1, 1]
            elif i == (len(self.refer_path)-1):
                dx = self.refer_path[i, 0] - self.refer_path[i-1, 0]
                dy = self.refer_path[i, 1] - self.refer_path[i-1, 1]
                ddx = self.refer_path[i, 0] + \
                    self.refer_path[i-2, 0] - 2*self.refer_path[i-1, 0]
                ddy = self.refer_path[i, 1] + \
                    self.refer_path[i-2, 1] - 2*self.refer_path[i-1, 1]
            else:
                dx = self.refer_path[i+1, 0] - self.refer_path[i, 0]
                dy = self.refer_path[i+1, 1] - self.refer_path[i, 1]
                ddx = self.refer_path[i+1, 0] + \
                    self.refer_path[i-1, 0] - 2*self.refer_path[i, 0]
                ddy = self.refer_path[i+1, 1] + \
                    self.refer_path[i-1, 1] - 2*self.refer_path[i, 1]
            self.refer_path[i, 2] = math.atan2(dy, dx)  # 计算每个路径点的航向角（yaw)
            # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
            # 参考：https://blog.csdn.net/weixin_46627433/article/details/123403726
            self.refer_path[i, 3] = (
                ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))  # 曲率k计算

    def calc_track_error(self, x, y):
        """计算跟踪误差

        Args:
            x (_type_): 当前车辆的位置x
            y (_type_): 当前车辆的位置y

        Returns:
            _type_: _description_
        """
        # 寻找参考轨迹最近目标点
        d_x = [self.refer_path[i, 0]-x for i in range(len(self.refer_path))]#计算每个路径点到当前车辆位置的x方向距离
        d_y = [self.refer_path[i, 1]-y for i in range(len(self.refer_path))]#计算每个路径点到当前车辆位置的y方向距离
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]#计算每个路径点到当前车辆位置的距离
        s = np.argmin(d)  # 因为d是一个列表,返回的是d最小的点的索引,也就是离车辆最近的点的索引

        yaw = self.refer_path[s, 2]#当前车辆参考位置的航向角
        k = self.refer_path[s, 3]#当前车辆参考位置的曲率
        angle = normalize_angle(yaw - math.atan2(d_y[s], d_x[s]))#计算当前车辆位置与最近参考路径点连线的角度和参考路径点的航向角的差值
        e = d[s]  # 欧几里得误差?
        if angle < 0:
            e *= -1

        return e, k, yaw, s

    def calc_ref_trajectory(self, robot_state, dl=1.0):
        """计算参考轨迹点，统一化变量数组，便于后面MPC优化使用
            参考自https://github.com/AtsushiSakai/PythonRobotics/blob/eb6d1cbe6fc90c7be9210bf153b3a04f177cc138/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
        Args:
            robot_state (_type_): 车辆的状态(x,y,yaw,v)
            dl (float, optional): _description_. Defaults to 1.0.参考路径点之间的间隔

        Returns:
            _type_: _description_
        """
        e, k, ref_yaw, ind = self.calc_track_error(#调用calc_track_error函数计算跟踪误差,
            robot_state[0], robot_state[1])#并返回跟踪误差，曲率，航向角，最近目标点的索引

        xref = np.zeros((NX, T + 1))#初始化参考轨迹点的状态数组 xref 和参考控制量数组 dref,NX是状态变量的维度
        dref = np.zeros((NU, T))#T 是预测时域长度。NU 是控制变量的维度。
        ncourse = len(self.refer_path)#获取路径点的总数

        xref[0, 0] = self.refer_path[ind, 0]#x坐标
        xref[1, 0] = self.refer_path[ind, 1]#y坐标
        xref[2, 0] = self.refer_path[ind, 2]#航向角
        #初始化参考轨迹点的状态数组

        # 参考控制量[v,delta]
        ref_delta = math.atan2(L*k, 1)#计算参考的转向角
        dref[0, :] = robot_state[3]#速度
        dref[1, :] = ref_delta#转向角

        travel = 0.0

        for i in range(T + 1):#计算未来的参考轨迹点并将其存储在xref数组中
            travel += abs(robot_state[3]) * dt#abs返回绝对值,这里是计算在当前时间步长内的行驶距离。
            dind = int(round(travel / dl))#round函数输出四舍五入的值后转化为整数,获得参考路径点的索引增量

            if (ind + dind) < ncourse:
                xref[0, i] = self.refer_path[ind + dind, 0]
                xref[1, i] = self.refer_path[ind + dind, 1]
                xref[2, i] = self.refer_path[ind + dind, 2]

            else:
                xref[0, i] = self.refer_path[ncourse - 1, 0]
                xref[1, i] = self.refer_path[ncourse - 1, 1]
                xref[2, i] = self.refer_path[ncourse - 1, 2]

        return xref, ind, dref


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    copied from https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/stanley_control/stanley_control.html
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def linear_mpc_control(xref, x0, delta_ref, ugv):
    """
    linear mpc control

    xref: reference point
    x0: initial state
    delta_ref: 参考输入
    ugv:车辆对象
    returns: 最优的控制量和最优状态
    """
    #定义优化变量
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0  # 代价函数
    constraints = []  # 约束条件

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t]-delta_ref[:, t], R)
        #将控制量的偏差加入代价函数
        if t != 0:
            cost += cvxpy.quad_form(x[:, t] - xref[:, t], Q)
        #将状态量的偏差加入代价函数
        A, B, C = ugv.state_space(delta_ref[1, t], xref[2, t])#根据参考量计算状态空间
        constraints += [x[:, t + 1]-xref[:, t+1] == A @
                        (x[:, t]-xref[:, t]) + B @ (u[:, t]-delta_ref[:, t])]


    cost += cvxpy.quad_form(x[:, T] - xref[:, T], Qf)#终端状态的代价函数

    constraints += [(x[:, 0]) == x0]#约束条件
    constraints += [cvxpy.abs(u[0, :]) <= MAX_VEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)#求解问题
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        """
        cvxpy.Problem.solve() 运行后，会返回 prob.status，表示求解器的状态。常见的 status 值包括：
        cvxpy.OPTIMAL：成功找到最优解 ✅。
        cvxpy.OPTIMAL_INACCURATE：找到的解可能是最优解，但存在 数值误差（可能是由于计算精度或条件数较差的优化问题）。
        cvxpy.INFEASIBLE：问题 无解 ❌，例如约束条件无法同时满足。
        cvxpy.UNBOUNDED：目标函数 没有界限，可能是约束条件不充分。
        其他状态（如 cvxpy.SOLVER_ERROR）：求解器失败。
        """
        opt_x = get_nparray_from_matrix(x.value[0, :])#get_nparray_from_matrix() 将 cvxpy 变量转换为 numpy 数组
        opt_y = get_nparray_from_matrix(x.value[1, :])
        opt_yaw = get_nparray_from_matrix(x.value[2, :])
        opt_v = get_nparray_from_matrix(u.value[0, :])
        opt_delta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = None, None, None, None, None,

    return opt_v, opt_delta, opt_x, opt_y, opt_yaw



def main():

    reference_path = MyReferencePath()
    goal = reference_path.refer_path[-1, 0:2]

    # 运动学模型
    ugv = KinematicModel_3(x_0, y_0, psi_0, v, L, dt)
    x_ = []
    y_ = []
    fig = plt.figure(1)
    # 保存动图用
    camera = Camera(fig)
    # plt.ylim([-3,3])
    for i in range(500):
        robot_state = np.zeros(4)
        robot_state[0] = ugv.x
        robot_state[1] = ugv.y
        robot_state[2] = ugv.psi
        robot_state[3] = ugv.v
        x0 = robot_state[0:3]
        xref, target_ind, dref = reference_path.calc_ref_trajectory(
            robot_state)
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = linear_mpc_control(
            xref, x0, dref, ugv)
        ugv.update_state(0, opt_delta[0])  # 加速度设为0，恒速

        x_.append(ugv.x)
        y_.append(ugv.y)

        # 显示动图
        plt.cla()
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:,
                 1], "-.b",  linewidth=1.0, label="course")
        plt.plot(x_, y_, "-r", label="trajectory")
        plt.plot(reference_path.refer_path[target_ind, 0],
                 reference_path.refer_path[target_ind, 1], "go", label="target")
        # plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

        # camera.snap()
        # 判断是否到达最后一个点
        if np.linalg.norm(robot_state[0:2]-goal) <= 0.1:
            print("reach goal")
            break
    # animation = camera.animate()
    # animation.save('trajectory.gif')


if __name__=='__main__':
    main()