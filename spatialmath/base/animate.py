#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:44:45 2020

@author: corkep
"""
#matplotlib inline

# line.set_data()
# text.set_position()
# quiver.set_offsets(), quiver.set_UVC()
# FancyArrow.set_xy()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import spatialmath.base as tr
import numpy as np
import math




    
class Animate:
    
    def __init__(self, axes=None, dims=None, projection='ortho', labels=['X', 'Y', 'Z'], **kwargs):
        self.displaylist = []
        
        if axes is None:
            # create an axes
            fig = plt.gcf()
            if fig.axes == []:
                # no axes in the figure, create a 3D axes
                axes = fig.add_subplot(111, projection='3d', proj_type=projection)
                axes.set_xlabel(labels[0])
                axes.set_ylabel(labels[1])
                axes.set_zlabel(labels[2])
                axes.autoscale(enable=True, axis='both')
            else:
                # reuse an existing axis
                axes = plt.gca()

        if dims is not None:
            if len(dims) == 2:
                dims = dims * 3
            axes.set_xlim(dims[0:2])
            axes.set_ylim(dims[2:4])
            axes.set_zlim(dims[4:6])
            #ax.set_aspect('equal')
                
        self.ax = axes
        
        #set flag for 2d or 3d axes, flag errors on the methods called later
        
    def draw(self, T):
        for x in self.displaylist:
            x.draw(T)
            
    def run(self, movie=None, axes=None, repeat=True, interval=50, nframes=100, **kwargs):


        def update(frame, a):
            s = frame/100.0;
            T = tr.transl(0.5*s, 0.5*s, 0.5*s) @ tr.trotx(math.pi*s)
            a.draw(T)
            return a.artists()
        
        # blit leaves a trail and first frame
        if movie is not None:
            repeat = False
        ani = animation.FuncAnimation(fig=plt.gcf(), func=update, frames=range(0,nframes), fargs=(self,), blit=False, interval=interval, repeat=repeat)
        
        if movie is None:
            plt.show()
        else:
            # Set up formatting for the movie files
            print('creating movie', movie)
             
             
            #plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
            
            FFwriter=animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
            ani.save(movie, writer=FFwriter)
            # TODO needs conda install -c conda-forge ffmpeg
            
    def __repr__(self):
        return ', '.join([x.type for x in self.displaylist])
    
    def artists(self):
        return [x.h for x in self.displaylist]
        
    #------------------- plot()

    class Line:
        
        def __init__(self, anim, h, xs, ys, zs):
            p = zip(xs, ys, zs)
            self.p = np.vstack([xs, ys, zs, [1,1]])
            self.h = h
            self.type = 'line'
            self.anim = anim
    
        def draw(self, T):
            p = T @ self.p
            self.h.set_data(p[0,:], p[1,:])
            self.h.set_3d_properties(p[2,:])
            

    def plot(self, xs, ys, zs, *args, **kwargs):
        h, = self.ax.plot(xs, ys, zs, *args, **kwargs)
        self.displaylist.append(Animate.Line(self, h, xs, ys, zs))
        
    #------------------- quiver()
   
    class Quiver:
        
        def __init__(self, anim, h):
            self.type = 'quiver'
            self.anim = anim
            # ._segments3d is 3x2x3
            #   first index: line segment in the collection
            #   second index: 0 = start, 1 = end
            #   third index: x, y, z components
            # https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib
            
            # turn to homogeneous form, with columns per point
            self.p = np.vstack( [h._segments3d.reshape(6,3).T, np.ones((1,6))])
            self.h = h
            self.type = 'arrow'
            self.anim = anim
                
        def draw(self, T):
            p = T @ self.p
    
            # reshape it
            p = (p[0:3,:].T).reshape(3,2,3)
            self.h.set_segments(p)
            
    def quiver(self, x, y, z, u, v, w, *args, **kwargs):
        h = self.ax.quiver(x, y, z, u, v, w, *args, **kwargs)
        self.displaylist.append(Animate.Quiver(self, h))

    #------------------- text()

    class Text:
        
        def __init__(self, anim, h, x, y, z):
            self.type = 'text'
            self.h = h
            self.p = np.r_[x, y, z, 1]
            self.anim = anim
    
        def draw(self, T):
            p = T @ self.p
            # x2, y2, _ = proj3d.proj_transform(p[0], p[1], p[2], self.anim.ax.get_proj())
            # self.h.set_position((x2, y2))
            self.h.set_position((p[0], p[1]))
            self.h.set_3d_properties(p[2])
        
        
    def text(self, x, y, z, *args, **kwargs):
        h = self.ax.text3D(x, y, z, *args, **kwargs)
        self.displaylist.append(Animate.Text(self, h, x, y, z))


    #------------------- scatter()

    def scatter(self, **kwargs):
        pass
            

    #------------------- wrappers for Axes primitives

    
    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)
        
    def set_ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)
        
    def set_zlim(self, *args, **kwargs):
        self.ax.set_zlim(*args, **kwargs)
        
    def set_xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)
        
    def set_ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)
        
    def set_zlabel(self, *args, **kwargs):
        self.ax.set_zlabel(*args, **kwargs)
    

        

def tranimate(T, **kwargs):
    anim = Animate(**kwargs)
    tr.trplot(T, axes=anim, **kwargs)
    anim.run(**kwargs)
    
    
tranimate( tr.transl(0,0,0), frame='A', arrow=False, dims=[0,5], movie='bob.mp4')






# a = trplot_a( tr.transl(1,2,3), frame='A', rviz=True, width=1)
# print(a)
# a.draw(tr.transl(0, 0, -1))
# trplot_a( tr.transl(3,1, 2), color='red', width=3, frame='B')
# trplot_a( tr.transl(4, 3, 1)@tr.trotx(math.pi/3), color='green', frame='c')
