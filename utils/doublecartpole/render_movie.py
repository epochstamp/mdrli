""" The script is slightly adapted from:

https://github.com/toddsifleet/inverted_pendulum/blob/master/render_movie.py

Author: Aaron Zixiao Qiu
"""

import os
import shutil
import numpy as np
from math import sin, cos, pi

from multiprocessing import Process
import matplotlib.pyplot as plt

PI = np.pi

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def save_mp4(data, n, video_prefix=""):
    if (not os.path.exists(video_prefix + '_imgs')):
        os.makedirs(video_prefix + '_imgs')

    if (not os.path.exists(video_prefix + '_video')):
        os.makedirs(video_prefix + '_video')

    # Create temporal layout at the bottom
    fig = plt.figure(0)
    fig.suptitle("Double Pendulum on Cart")

    cart_time_line = plt.subplot2grid(
        (12, 14),
        (9, 0),
        colspan=12,
        rowspan=6
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

    # Draw theta 1 curve
    pendulum_time_line = cart_time_line.twinx()
    pendulum_time_line.axis([
        0,
        t_max,
        -PI,
        PI
    ])
    pendulum_time_line.set_ylabel('theta 1 (rad)')
    pendulum_time_line.plot(data[:,0], data[:,3],'g-')

    # Draw theta 2 curve
    pendulum_time_line = cart_time_line.twinx()
    pendulum_time_line.set_ylabel('theta 2 (rad)')
    pendulum_time_line.spines["right"].set_position(("axes", 1.2))
    pendulum_time_line.spines["right"].set_visible(True)
    make_patch_spines_invisible(pendulum_time_line)
    
    pendulum_time_line.axis([
        0,
        t_max,
        -PI,
        PI
    ])
    pendulum_time_line.plot(data[:,0], data[:,5],'-', color='orange')

    # Cart layout
    cart_plot = plt.subplot2grid(
        (12,14),
        (0,0),
        rowspan=8,
        colspan=12
    )
    cart_plot.axes.get_yaxis().set_visible(False)

    # Draw cart and pole
    t = 0
    fps = 25.
    frame_number = 1
    x_min = min([min(data[:,1]), -1.1])
    x_max = max([max(data[:,1]), 1.1])

    time_bar, = cart_time_line.plot([0,0], [10000, -10000], lw=3)

    for point in data:
        if point[0] >= t + 1./fps or not t:
            _draw_point(point, time_bar, t, x_min, x_max, cart_plot)
            t = point[0]
            fig.savefig(video_prefix + '_imgs/_tmp%03d.png' % frame_number)
            frame_number += 1

    
    os.system("ffmpeg -y -framerate 25 -i "+video_prefix+"_imgs/_tmp%03d.png  -c:v libx264 -pix_fmt yuv420p "+video_prefix+"_video/_out" + str(n) + ".mp4 > /dev/null 2> /dev/null")
	
    shutil.rmtree(video_prefix+"_imgs/")
    return

def _draw_point(point, time_bar, t, x_min, x_max, cart_plot):
    # Draw cart
    time_bar.set_xdata([t, t])
    cart_plot.cla()
    cart_plot.axis([x_min,x_max,-.5,.5])
    l_cart = 0.05 * (x_max + abs(x_min))
    cart_plot.plot([point[1]-l_cart,point[1]+l_cart], [0,0], 'r-', lw=5)

    # Draw pole
    theta = point[3] 
    x = sin(theta)
    y = cos(theta)
    theta2 = point[5] 
    x2 = sin(theta2)
    y2 = cos(theta2)
    l_pole = 0.2 * (x_max + abs(x_min))
    cart_plot.plot([point[1],point[1]+l_pole*x],[0,.4*y],'g-', lw=4)
    cart_plot.plot([point[1]+l_pole*x,point[1]+l_pole*x-l_pole*x2],[.4*y,.4*y + .4*y2],'-',color='orange', lw=4)
    
    return 
