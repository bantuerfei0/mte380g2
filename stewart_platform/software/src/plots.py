"""Render the two experiment panels with OpenCV for display or PNG export."""
import cv2
import numpy as np

from .display import Window, label


def response_plot(trials, measured, show, path):
    colors = [(205,112,45), (80,150,30), (45,105,210), (160,75,165), (140,140,45)]
    image = np.full((660, 980, 3), 249, np.uint8)
    dark = (50,50,50)
    end = max(.001, max(data[-1,0] for _,data in trials))
    for panel in range(2):
        curves = []
        for i,(name,data) in enumerate(trials):
            if measured or int(i >= 3) == panel:
                axis = panel if measured else 0
                curves.append((name, data[:,0], data[:,1+axis]-data[:,3+axis], colors[i%len(colors)]))
        title = ("XY"[panel]+" position error") if measured else ("Damping and midpoint" if panel == 0 else "Homing and IMU feedback")
        top, bottom = 43+panel*320, 268+panel*320
        left, right = 70, 710
        low = min(0., min(values.min() for _,_,values,_ in curves))
        high = max(0., max(values.max() for _,_,values,_ in curves))
        pad = max(.1, .06*(high-low))
        low, high = low-pad, high+pad
        label(image, title+" (cm)", (left,top-12), dark)
        for fraction in np.linspace(0,1,6):
            y = round(bottom-fraction*(bottom-top))
            x = round(left+fraction*(right-left))
            cv2.line(image,(left,y),(right,y),(220,220,220))
            cv2.line(image,(x,top),(x,bottom),(220,220,220))
            label(image,f"{low+fraction*(high-low):.1f}",(4,y+4),dark)
            label(image,f"{fraction*end:.1f}",(x-9,bottom+19),dark)
        label(image,"Time (s)",((left+right)//2-25,bottom+39),dark)
        for i,(name,t,values,color) in enumerate(curves):
            points = np.column_stack((left+t/end*(right-left), bottom-(values-low)/(high-low)*(bottom-top)))
            cv2.polylines(image,[np.rint(points).astype(np.int32)],False,color,2,cv2.LINE_AA)
            label(image,name,(right+12,top+18*i+12),color)
    if path and not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write plot to {path}")
    if show:
        window = Window("Experiment response")
        try:
            while True:
                button, _, key = window.poll(image, ["Back"])
                if button or key in (27,13,ord("q")):
                    break
        finally:
            window.close()
