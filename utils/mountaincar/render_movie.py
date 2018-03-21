""" The script is slightly adapted from:

https://github.com/toddsifleet/inverted_pendulum/blob/master/render_movie.py

Author: Aaron Zixiao Qiu
"""

import os
import numpy as np
from math import sin, cos, pi

from multiprocessing import Process
import matplotlib.pyplot as plt

PI = np.pi

def save_mp4(env,data, n, video_prefix=""):
    if (not os.path.exists(video_prefix + '_imgs')):
        os.makedirs(video_prefix + '_imgs')

    if (not os.path.exists(video_prefix + '_video')):
        os.makedirs(video_prefix + '_video')
        
    fig = plt.figure(0)
    fig.suptitle("Mountain car")
    
    cart_time_line = plt.subplot2grid(
        (12, 12),
        (9, 0),
        colspan=12,
        rowspan=3
    )
    
    # Draw displacement curve
    t_max = max(data[:,0])
    cart_time_line.axis([
        0,
        t_max,
        min(data[:,1])*1.1,
        max(data[:,1])*1.1+.1,
    ])
    cart_time_line.set_xlabel('time (s)')
    cart_time_line.set_ylabel('x (m)')
    cart_time_line.plot(data[:,0], data[:,1],'r-')
    
    
    # Draw theta curve
    pendulum_time_line = cart_time_line.twinx()
    pendulum_time_line.axis([
        0,
        t_max,
        min(data[:,2])*1.1-.1,
        max(data[:,2])*1.1
    ])
    pendulum_time_line.set_ylabel('speed (m/s)')
    pendulum_time_line.plot(data[:,0], data[:,2],'g-')
    
    
    # Cart layout
    cart_plot = plt.subplot2grid(
        (12,12),
        (0,0),
        rowspan=8,
        colspan=12
    )
    cart_plot.axes.get_yaxis().set_visible(False)
    
    # Draw cart and pole
    t = 0
    fps = 25.
    frame_number = 1
    
    time_bar, = cart_time_line.plot([0,0], [10000, -10000], lw=3)
    for point in data:
        if point[0] >= t + 1./fps or not t:
            _draw_point(env,point, time_bar, t, cart_plot)
            t = point[0]
            fig.savefig(video_prefix + '_imgs/_tmp%03d.png' % frame_number)
            frame_number += 1
    
    print(os.system("ffmpeg -framerate 25 -i "+video_prefix+"_imgs/_tmp%03d.png  -c:v libx264 -pix_fmt yuv420p "+video_prefix+"_video/_out" + str(n) + ".mp4 > /dev/null 2> /dev/null"))
                
def _height(self, xs):
    return np.sin(3 * xs)*.45+.55
def _draw_point(env,point, time_bar, t, cart_plot):
    # Draw cart
    time_bar.set_xdata([t, t])
    cart_plot.cla()
    curve = np.arange(env.min_position,env.max_position,abs(env.max_position-env.min_position)/100)
    height = env._height(curve);
    cart_plot.axis([env.min_position,env.max_position,min(height)-0.1,max(height)+0.1])

    cart_plot.plot(point[1],env._height(point[1]),'ro')
    cart_plot.plot(curve,env._height(curve),'b')

    return 