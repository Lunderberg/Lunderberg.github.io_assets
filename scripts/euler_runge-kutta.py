#!/usr/bin/env python

import numpy
import pylab

class Eulers(object):
    def __init__(self, initial_state, derivative, step_size, initial_time = 0):
        self.state = initial_state
        self.der = derivative
        self.step_size = step_size
        self.time = initial_time

    def Step(self):
        self.state += self.step_size * self.der(self.state, self.time)
        self.time += self.step_size


class RungeKutta(object):
    def __init__(self, initial_state, derivative, step_size, initial_time = 0):
        self.state = initial_state
        self.der = derivative
        self.step_size = step_size
        self.time = initial_time

    def Step(self):
        k1 = self.step_size * self.der(self.state, self.time)
        k2 = self.step_size * self.der(self.state + k1/2.0, self.time + self.step_size/2.0)
        k3 = self.step_size * self.der(self.state + k2/2.0, self.time + self.step_size/2.0)
        k4 = self.step_size * self.der(self.state + k3, self.time + self.step_size)
        self.state += (k1 + 2*k2 + 2*k3 + k4)/6.0
        self.time += self.step_size


def plot_x2():
    tvals = numpy.arange(0,5,0.01)
    xvals = tvals*tvals

    euler = Eulers(0.0, lambda x,t : 2*t, 1.0)
    tvals_euler = []
    xvals_euler = []
    while euler.time <= tvals.max():
        tvals_euler.append(euler.time)
        xvals_euler.append(euler.state)
        euler.Step()
    tvals_euler.append(euler.time)
    xvals_euler.append(euler.state)


    rk4 = RungeKutta(0.0, lambda x,t : 2*t, 1.0)
    tvals_rk4 = []
    xvals_rk4 = []
    while rk4.time <= tvals.max():
        tvals_rk4.append(rk4.time)
        xvals_rk4.append(rk4.state)
        rk4.Step()
    tvals_rk4.append(rk4.time)
    xvals_rk4.append(rk4.state)


    pylab.plot(tvals, xvals, label='Exact')
    pylab.plot(tvals_euler, xvals_euler, label="Euler's Method")
    pylab.legend(loc='best')
    #pylab.show()
    pylab.savefig('../math/eulers_method_x2.png')
    pylab.close()

    pylab.plot(tvals, xvals, label='Exact')
    pylab.plot(tvals_euler, xvals_euler, label="Euler's Method")
    pylab.plot(tvals_rk4, xvals_rk4, label="Runge-Kutta")
    pylab.legend(loc='best')
    #pylab.show()
    pylab.savefig('../math/runge_kutta_x2.png')
    pylab.close()




def plot_trigometric():
    tvals = numpy.arange(0,7,0.01)
    xvals = numpy.cos(tvals)

    initial = numpy.array([1.0, 0.0])
    derivative = lambda x,t : numpy.array([-x[1],x[0]])
    step_size = 0.25

    euler = Eulers(numpy.array(initial), derivative, step_size)
    tvals_euler = []
    xvals_euler = []
    while euler.time <= tvals.max():
        tvals_euler.append(euler.time)
        xvals_euler.append(euler.state[0])
        euler.Step()
    tvals_euler.append(euler.time)
    xvals_euler.append(euler.state[0])

    rk4 = RungeKutta(numpy.array(initial), derivative, step_size)
    tvals_rk4 = []
    xvals_rk4 = []
    while rk4.time <= tvals.max():
        tvals_rk4.append(rk4.time)
        xvals_rk4.append(rk4.state[0])
        rk4.Step()
    tvals_rk4.append(rk4.time)
    xvals_rk4.append(rk4.state[0])

    pylab.plot(tvals, xvals, label='Exact')
    pylab.plot(tvals_euler, xvals_euler, label="Euler's Method")
    pylab.plot(tvals_rk4, xvals_rk4, label="Runge-Kutta")
    pylab.legend(loc='best')
    #pylab.show()
    pylab.savefig('../math/runge_kutta_trig.png')
    pylab.close()


if __name__=='__main__':
    plot_x2()
    plot_trigometric()
