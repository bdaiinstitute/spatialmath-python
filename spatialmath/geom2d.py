#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 09:42:30 2020

@author: corkep
"""

from spatialmath import base
import matplotlib.pyplot as plt
import numpy as np

# geom2d
def plot_line():
    pass

def plot_box(ax=None, 
        bbox=None, bl=None, tl=None, br=None, tr=None, wh=None, centre=None,
        color=None, fillcolor=None, alpha=None, thickness=None, **kwargs):
    """
    Plot a box using matplotlib

    :param ax: the axes to draw on, defaults to ``gca()``
    :type ax: Axis, optional
    :param bbox: bounding box matrix, defaults to None
    :type bbox: ndarray(2,2), optional
    :param bl: bottom-left corner, defaults to None
    :type bl: array_like(2), optional
    :param tl: top-left corner, defaults to None
    :type tl: [array_like(2), optional
    :param br: bottom-right corner, defaults to None
    :type br: array_like(2), optional
    :param tr: top -ight corner, defaults to None
    :type tr: array_like(2), optional
    :param wh: width and height, defaults to None
    :type wh: array_like(2), optional
    :param centre: [description], defaults to None
    :type centre: array_like(2), optional
    :param color: box outline color
    :type color: array_like(3) or str
    :param fillcolor: box fill color
    :type fillcolor: array_like(3) or str
    :param alpha: transparency, defaults to 1
    :type alpha: float, optional
    :param thickness: line thickness, defaults to None
    :type thickness: float, optional
    :return: the matplotlib object
    :rtype: Patch.Rectangle

    Plots a box on the specified axes using matplotlib

    The box can be specified in many ways:

    - bounding box which is a 2x2 matrix [xmin, xmax; ymin, ymax]
    - centre and width+height
    - bottom-left and top-right corners
    - bottom-left corner and width+height
    - top-right corner and width+height
    - top-left corner and width+height
    """

    if bbox is not None:
        xy = bbox[:,0]
        w = bbox[0,1] - bbox[0,0]
        h = bbox[1,1] - bbox[1,0]
    elif bl is not None and tl is None and tr is None and wh is not None and centre is None:
        # bl + wh
        xy = bl
        w, h = wh
    elif bl is not None and tl is None and tr is not None and wh is None and centre is None:
        # bl + tr
        xy = bl
        w = br[0] - bl[0]
        h = br[1] - bl[1]
    elif bl is None and tl is None and tr is None and wh is not None and centre is not None:
        # centre + wh
        w, h = wh
        xy = (centre[0] - w / 2, centre[1] - h / 2)
    elif bl is None and tl is None and tr is not None and wh is not None and centre is None:
        # tr + wh
        w, h = wh
        xy = (tr[0] - wh[0], tr[1] - wh[1])
    elif bl is None and tl is not None and tr is None and wh is not None and centre is None:
        # tl + wh
        w, h = wh
        xy = (tl[0], tl[1] - h)

    if ax is None:
        ax = plt.gca()

    fill = fillcolor is not None
    rect = plt.Rectangle(xy, w, h, edgecolor=color, facecolor=fillcolor, fill=fill,
    alpha=alpha, linewidth=thickness, clip_on=True)
    ax.add_patch(rect)
    plt.draw()

    return rect




def plot_text(pos, text=None, ax=None, color=None, **kwargs):
    """
    Plot text using matplotlib

    :param pos: position of text
    :type pos: array_like(2)
    :param text: text
    :type text: str
    :param ax: axes to draw in, defaults to ``gca()``
    :type ax: Axis, optional
    :param color: text color, defaults to None
    :type color: str or array_like(3), optional
    :param kwargs: additional arguments passed to ``pyplot.text()``
    """
    
    defaults = {
        'horizontalalignment': 'left',
        'verticalalignment': 'center'
    }
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v
    if ax is None:
        ax = plt.gca()
    plt.text(pos[0], pos[1], text, color=color, **kwargs)




def plot_point(pos, marker='bs', text=None, ax=None, color=None, textargs=None, **kwargs):
    """
    Plot a point using matplotlib

    :param pos: position of marker
    :type pos: array_like(2), ndarray(2,n), list of 2-tuples
    :param marker: matplotlub marker style, defaults to 'bs'
    :type marker: str or list of str, optional
    :param text: text label, defaults to None
    :type text: str, optional
    :param ax: axes to plot in, defaults to ``gca()````
    :type ax: Axis, optional
    :param color: text color, defaults to None
    :type color: str or array_like(3), optional

    The color of the marker can be different to the color of the text,
    the marker color is specified by a single letter in the marker string.

    A point can multiple markers which will be overlaid, for instance ``["rx",
    "ro"]`` will give a ⨂ symbol.

    The optional text label is placed to the right of the marker, and vertically
    aligned. 
    
    Multiple points can be marked if ``pos`` is a 2xn array or a list of
    coordinate pairs.  If a label is provided every point will have the same
    label. However, the text is processed with ``format`` and is provided with a
    single argument, the point index (starting at zero).


    """
    
    if isinstance(pos, np.ndarray) and pos.shape[0] == 2:
        x = pos[0,:]
        y = pos[1,:]
    elif isinstance(pos, (tuple, list)):
        # [x, y]
        # [(x,y), (x,y), ...]
        # [xlist, ylist]
        # [xarray, yarray]
        if base.islistof(pos, (tuple, list)):
            x = [z[0] for z in pos]
            y = [z[1] for z in pos]
        elif base.islistof(pos, np.ndarray):
            x = pos[0]
            y = pos[1]
        else:
            x = pos[0]
            y = pos[1]

    textopts = {'fontsize': 12}
    if textargs is not None:
        textopts = {**textopts, **textargs}

    if ax is None:
        ax = plt.gca()
    if isinstance(marker, (list, tuple)):
        for m in marker:
            plt.plot(x, y, m, **kwargs)
    else:
        plt.plot(x, y, marker)
    if text:
        try:
            for i, xy in enumerate(zip(x, y)):
                plt.text(xy[0], xy[1], ' ' + text.format(i), horizontalalignment='left', verticalalignment='center', color=color, **textopts)
        except:
            plt.text(x, y, ' ' + text, horizontalalignment='left', verticalalignment='center', color=color, **textopts)



    
    
def intersect_lines():
    # return mindist and points if no intersect
    pass

def distance_line_line():
    pass

def distance_line_point():
    pass

def points_join():

    pass

def intersect_polygon___line():
    pass

def contains_polygon_point():
    pass