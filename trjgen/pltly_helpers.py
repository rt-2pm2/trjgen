#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions to plot with plotly/dash 

@author: rt-2pm2
"""

import plotly as py
import plotly.graph_objs as go


def TrajFromPW_plotly(X, Y, Z, t):
    trace1 = go.Scatter3d(
            x = X[0, :],
            y = Y[0, :],
            z = Z[0, :],
            mode = 'markers',
            marker = dict(
                size = 3,
                color = t,
                colorscale = 'Viridis',
                opacity = 0.8
                )
            )

    data = [trace1]
    layout = go.Layout(
            margin = dict(
                l = 0,
                r = 0,
                b = 0,
                t = 0
                )
            )
    fig = go.Figure(data = data, layout = layout)
    py.offline.plot(fig, filename='3D Trajectory')
