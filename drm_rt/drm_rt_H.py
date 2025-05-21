#! /usr/env/bin python3
import os
import csv
import time
import rospy
import numpy as np
import datetime as dt
import threading as th
import xlsxwriter

## Helper functions
import motor
import linear_cf

from itertools import islice
from ceilingEffect import thrustCE
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from std_msgs.msg import String, Float32, Int16

tnrfont = {'fontname':'Times New Roman'}
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
xs = []
ys = []
log = []
max_wr = 0
keep_going = True 


def key_capture_thread():
  global keep_going
  input()
  keep_going = False


class drmRT():
  def __init__(self, selection):
    self.init_params(selection)
    self.init_nodes()


  def init_params(self, selection):
    global max_wr, keep_going
    ''' Topic related '''
    self.cdist = 1.0
    self.rpm = 3000
    self.current = 0.01
    self.UAVthrust = 0
    self.initialRpm = 0
    self.currDrill = 0.01
    self.voltDrill = 0.0

    ''' System params '''
    self.selection = selection
    # self.force_ce = 0
    self.K_spG = 0.1
    self.K_spL = 0.2
    self.mg = 750/1000 * 9.81
    self.F_total = 0
    self.sp_L = (0.08-self.cdist)*self.K_spL
    self.sp_G = (0.08-self.cdist)*self.K_spG
    self.zeta = 0
    self.epsilon = 0.95
    self.sigma = 0.0025
    self.lamb = 0.0001
    self.miu = 1
    self.zeta = np.tan(np.deg2rad(40))
    self.ncutter = 2
    # self.motorThrust = motor.getThrust()
    # self.motorThrottle = motor.getThrottle()
    # self.motorCurrent = motor.getCurrent()
    self.depthCut = 0
    self.matThickness = 10    # mm
    self.weightBit = 0
    self.widthbit = 0.003
    self.norm_wr = 0
    self.max_wr = max_wr
    self.F_feed = 0
    self.F_drill = 0
    self.F_mgx = 0
    self.tau_slip = 0.197
    self.dm1 = 10.67/1000
    self.dm2 = 17.82/1000
    self.l1 = 1.25/1000
    self.l2 = 1.25/1000
    self.mu_t = 0.8

    ''' List initialization '''
    self.saveList = []
    self.timeList = []
    self.thrustEstList = []
    self.rpmList = []
    self.timeDurList = []
    self.timeStart = time.time()
    self.currList = []
    self.thrustList = []
    self.penedepth = 0.0
    self.penedepthList = []
    self.penetimeList = []
    self.voltDrillList = []
    self.currDrillList = []


  def init_nodes(self):
    rospy.init_node('drmRT')
    self.rate = rospy.Rate(100)
    self.init_pubsub()
    self.run_node()
    self.rate.sleep()


  def init_pubsub(self):
    # rospy.Subscriber('/ranger', String, self.ranger_callback)
    rospy.Subscriber('/ranger_pub', Float32, self.ranger_callback)
    rospy.Subscriber('/rpm_pub', Int16, self.rpm_callback)
    rospy.Subscriber('/current_pub', Float32, self.current_callback)
    rospy.Subscriber('/currDrill_pub', Float32, self.currDrill_callback)
    rospy.Subscriber('/voltDrill_pub', Float32, self.voltDrill_callback)
    # rospy.Subscriber('/throttle', Float32, self.thrust_callback)
    self.wr_pub = rospy.Publisher('/wbr', Float32, queue_size=1)
    self.wr_msg = Float32()


  def ranger_callback(self, msg):
    if msg.data == 0:
      self.cdist = 1.0
    else:
      self.cdist = msg.data
    rospy.loginfo('ranger: {0:.2f} m'.format(self.cdist))


  def rpm_callback(self, msg):
    if msg.data == 1:
      self.rpm = 2000
    else:
      self.rpm = msg.data
    rospy.loginfo('rpm: {0:.2f}'.format(self.rpm))


  # def current_callback(self, msg):
  #   '''
  #   // Scaling for power brick current sensing with arduino
  #   Scaling done under topicManager script
  #   '''
  #   # self.current = (19.41*msg.data)-3.6384  
  #   self.current = msg.data * 0.375
  #   rospy.loginfo('current: {0:.2f} A'.format(self.current))


  # def thrust_callback(self, msg):
  #   self.UAVthrust = msg.data
  #   rospy.loginfo('throttle: {0:.2f}'.format(self.UAVthrust))

  
  def currDrill_callback(self, msg):
    self.currDrill = msg.data
    rospy.loginfo('current_drill: {0:.2} A'.format(self.currDrill))

  
  def voltDrill_callback(self, msg):
    self.voltDrill = msg.data
    rospy.loginfo('voltage_drill: {0:.2} A'.format(self.voltDrill))


  # def ce_thrust(self):
  #   ''' to calculate ceiling effect '''
  #   self.force_ce = thrustCE(self.cdist).getThrust()
  #   # print("F_ce: ", self.force_ce)


  # def getMotorThrust(self):
  #   ''' get motor current in realtime, convert to thrust '''
  #   ## ''' get motor throttle in realtime, convert to thrust '''
  #   self.fit_m, self.fit_c = linear_cf.solveLinear(self.motorThrust, self.motorCurrent)
  #   self.thrustEst = linear_cf.linear_roots(self.current, self.fit_m, self.fit_c) * (4*9.81/1000) - self.mg   # for current-to-thrust conversion with datasheet
  #   # self.thrustEst = linear_cf.linear_roots(self.UAVthrust, self.fit_m, self.fit_c) 


  # def getTotalThrust(self):
  #   ''' sum of z-forces on UAV '''
  #   self.getMotorThrust()       # call prior function
  #   print('MotorThrust', self.thrustEst)
  #   print('Current: ', self.current)
  #   self.ce_thrust()
  #   if self.selection == 1:     # ceiling
  #     self.F_ztotal = self.thrustEst + self.force_ce - self.sp_L
  #   else:                       # beam
  #     self.F_ztotal = self.thrustEst - (self.sp_L + self.sp_G)

  def drill_feedforce(self):
    ''' Drill feedforce '''
    self.F_drill = (2*self.tau_slip/self.dm1)*((np.pi*self.dm1 - self.mu_t*self.l1)/(self.l1+np.pi*self.mu_t*self.dm1)) + (2*self.tau_slip/self.dm2)*((np.pi*self.dm2 - self.mu_t*self.l2)/(self.l2+np.pi*self.mu_t*self.dm2))
    self.F_feed = self.F_drill + self.F_mgx
    return self.F_feed


  def depthofCut(self):
    ''' get depth of cut '''
    self.getTotalThrust()       # call prior function
    ''' To get feedforce'''     ## TODO: get feedforce
    if (2.0*9.81) <= self.F_ztotal < (3.0*9.81):
      self.cutTime = 1.266
    elif (3.0*9.81) <= self.F_ztotal < (4.0*9.81):
      self.cutTime = 0.833
    else: self.cutTime = 0.48
    self.depthCut = (self.matThickness/self.cutTime)/self.rpm


  def drm(self):
    ''' for drm '''
    self.depthofCut()           # call prior function
    print('self.max_wr: ', self.max_wr)
    # counter = 0
    if self.cdist > 0.07 :
      self.wr = 0.0
      self.max_wr = 0.01
    else:
      self.penedepth = self.cdist
      self.weightBit = (self.zeta*self.epsilon*self.depthCut) + (self.ncutter*self.sigma*self.lamb)
      self.wr = self.weightBit/self.widthbit
      if self.wr > self.max_wr:
      # if self.wr > self.max_wr and (counter < 5):
        self.max_wr = self.wr
        # counter = counter + 1
      # elif counter >= 5:
      #   self.max_wr = 0.01
      #   counter = 0
    self.norm_wr = round((self.wr / self.max_wr), 3)
    self.wr_msg.data = self.norm_wr
    # self.wr_msg.data = self.wr
    self.wr_pub.publish(self.wr_msg)
    self.time_now = time.time()
    self.time_dur = self.time_now - self.timeStart
    self.dataSave()
    print('wr:', self.wr)
    print('max_wr: ', self.max_wr)
    print('self.norm_wr', self.norm_wr)


  def movingaverage(self, interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

  def dataSave(self):
    self.saveList.append(self.norm_wr)
    self.timeDurList.append(self.time_dur)
    self.timeList.append(dt.datetime.now().strftime('%H:%M:%S'))
    self.thrustEstList.append(self.thrustEst)
    self.rpmList.append(self.rpm)
    self.currList.append(float(self.current))
    self.currentList = [(float(i)/float(max(self.currList))) for i in self.currList]
    self.thrustList.append(float(self.F_ztotal))
    if self.penedepth == 0.0:
      self.penedepthList.append(float(self.penedepth))
    else: self.penedepthList.append(0.07-float(self.penedepth))
    # for i in range(len(self.timeDurList)):
    #   if 0 < i <= 1/3*len(self.timeDurList):
    #     self.voltDrillList.append((4.2*5)-self.voltDrill)
    #   elif 1/3*len(self.timeDurList) < i <= 2/3*len(self.timeDurList):
    #     self.voltDrillList.append((4.2*5)-self.voltDrill-(0.041667*3))
    #   elif 2/3*len(self.timeDurList) < i <= len(self.timeDurList): 
    #     self.voltDrillList.append((4.2*5)-self.voltDrill-(0.041667*6))
    #   else: 
    #     self.voltDrillList.append((4.2*5)-self.voltDrill-(0.041667*6))
    self.voltDrillList.append((4.2*5)-self.voltDrill)
    self.currDrillList.append(self.currDrill)

    window = 5
    self.avgVoltDrill = []
    self.avgCurrDrill = []
    self.avgTimeList = []
    # for ind in range(len(self.voltDrillList) - window + 1):
    #   self.avgVoltDrill.append(np.mean(self.voltDrillList[ind:ind+window]))
    #   self.avgCurrDrill.append(np.mean(self.currDrillList[ind:ind+window]))
    # for ind in range(window - 1):
    #   self.avgVoltDrill.insert(0, np.nan)
    #   self.avgCurrDrill.insert(0, np.nan)
    # self.avgVoltDrill = self.movingaverage(self.voltDrillList, 10)
    # self.avgCurrDrill = self.movingaverage(self.currDrillList, 10)
    # self.thrustListPlot = [(float(i)/float(max(self.thrustList))) for i in self.thrustList]


  def imgPlotter(self):
    plt.figure()
    # plt.plot(self.timeList, self.saveList)
    # self.normSave = [(float(i)/max(self.saveList)) for i in self.saveList]
    plt.plot(self.timeDurList, self.saveList)
    plt.ylabel('Normalized resistance')
    # plt.xlabel('Time (HH:MM:SS)')
    plt.xlabel('Duration of operation [secs]')
    plt.title('Resistogram_rt')
    plt.grid(axis='y')
    plt.savefig('Resistogram_rt.svg', dpi=600)
    
    
    # plt.figure()
    fig, axs = plt.subplots(3, figsize=(8,8))
    axs[0].plot(self.timeDurList, self.thrustList, label='motor_thrust [N]')
    axs[0].set_title('Feed force [N]')
    axs[0].grid(axis='y')
    axs[0].legend(loc='upper right')
    axs[1].plot(self.timeDurList, self.rpmList, 'tab:orange', label='RPM')
    axs[1].set_title('RPM')
    axs[1].grid(axis='y')
    axs[1].legend(loc='upper right')
    axs[2].plot(self.timeDurList, self.penedepthList, 'tab:green', label='Penetration [m]')
    axs[2].grid(axis='y')
    axs[2].set_title('Penetration Depth [m]')
    axs[2].legend(loc='upper right')

    for ax in axs.flat:
      ax.set(xlabel='Duration of operation [secs]')
      ax.label_outer()
    plt.savefig('Parameters_rt.svg', dpi=600)


    plt.figure()
    # plt.plot(self.timeDurList, self.currDrillList, label='drill_current_orig [A]')
    # print(len(self.voltDrillList))
    mult_List = np.linspace((1), (0.95), len(self.voltDrillList))
    comb_list = np.multiply(self.voltDrillList, mult_List)
    plt.plot(self.timeDurList, comb_list, label='drill_volt [V]')
    plt.plot(self.timeDurList, self.currDrillList, label='drill_current [A]')
    # plt.axhline(y=np.nanmean(self.voltDrillList), color='green')
    plt.axhline(y=np.nanmean(comb_list), color='green')
    plt.axhline(y=np.nanmean(self.currDrillList), color='green')


    plt.xlabel('Duration of operation [secs]')
    plt.grid(axis='y')
    plt.title('Drill motor params')
    plt.legend(loc='upper right')
    plt.savefig('Drill_motor_params.svg', dpi=600)
    plt.show()

    with open('data.csv', 'w') as k:
      writer = csv.writer(k)
      for i in range(len(self.saveList)):
        writer.writerow([self.saveList[i], self.timeList[i], self.thrustEstList[i], self.rpmList[i]]) 


  def plotter(self):
    """ for plotting """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ani = animation.FuncAnimation(fig, self.animate, fargs=(xs, ys), interval=1000)
    plt.show()


  def run_node(self):
    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    self.timeStart = time.time()
    while keep_going:
      self.drm()
      try:
        self.rate.sleep()
      except rospy.ROSInterruptException:
        pass
    self.imgPlotter()


if __name__ == "__main__":
  val = float(input("Select 1: ceiling or 2: beam? \n"))
  try:
    # time.sleep(2)
    drmRT(val)
    # rospy.spin()
  except rospy.ROSInterruptException:
    pass